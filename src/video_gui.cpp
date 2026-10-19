/*
 * bc-video-gui -- front end for the offline video pass.
 *
 * Deliberately a separate binary from betacensor-gui, and separate from the
 * live censor, because it is a different kind of job: the live GUI edits
 * settings that take effect immediately on a running process, while this one
 * starts a batch that runs to completion and produces a file. Putting a
 * long-running job with a progress bar into a live settings panel would mean
 * one window whose controls sometimes apply now and sometimes do not.
 *
 * What it does *not* do is duplicate the settings. Effect, classes, confidence
 * and caption all come from betacensor.conf -- the same file the live GUI
 * edits -- so an effect set up and eyeballed on screen is exactly the effect
 * written to the file. This window only asks the three questions that are
 * specific to a batch run: which file, where to put it, and how to encode it.
 *
 * No Q_OBJECT, so there is no moc step: every connection is a lambda.
 */

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QVBoxLayout>
#include <QWidget>
#include <QMessageBox>

namespace {

/*
 * Where the binaries are. The GUI is normally launched from the build tree
 * beside bc-video, but an absolute path next to this executable also works, so
 * a shortcut with a different working directory does not silently fail.
 */
QString videoBinary()
{
	const QString beside =
	    QCoreApplication::applicationDirPath() + "/bc-video";
	if (QFileInfo::exists(beside))
		return beside;
	return "build/bc-video";
}

/* "in.mkv" -> "in-censored.mp4". Never the input path itself: this program
 * writes its output while reading the input, and overwriting the source would
 * destroy the original halfway through the job. */
QString suggestOutput(const QString &in)
{
	QFileInfo fi(in);
	return fi.dir().filePath(fi.completeBaseName() + "-censored.mp4");
}

}  // namespace

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	QWidget win;
	win.setWindowTitle("BetaCensor — video file");
	auto *root = new QVBoxLayout(&win);

	/* ---- files ---- */
	auto *files = new QGroupBox("Files");
	auto *form  = new QFormLayout(files);

	auto *inEdit  = new QLineEdit;
	auto *inBrowse = new QPushButton("Browse…");
	auto *inRow   = new QHBoxLayout;
	inRow->addWidget(inEdit, 1);
	inRow->addWidget(inBrowse);
	form->addRow("Input", inRow);

	auto *outEdit = new QLineEdit;
	auto *outBrowse = new QPushButton("Browse…");
	auto *outRow  = new QHBoxLayout;
	outRow->addWidget(outEdit, 1);
	outRow->addWidget(outBrowse);
	form->addRow("Output", outRow);

	auto *info = new QLabel("—");
	info->setWordWrap(true);
	form->addRow("Source", info);

	root->addWidget(files);

	/* ---- encoding ---- */
	auto *encBox = new QGroupBox("Encoding");
	auto *encForm = new QFormLayout(encBox);
	auto *quality = new QComboBox;
	/*
	 * Presets rather than a free-text flag box. The data is the ffmpeg
	 * argument string, passed through as BC_VIDEO_ENC.
	 */
	quality->addItem("Good quality, slower (x264 CRF 18)",
	                 "-c:v libx264 -crf 18 -preset medium");
	quality->addItem("Balanced (x264 CRF 18, fast)",
	                 "-c:v libx264 -crf 18 -preset veryfast");
	quality->addItem("Smaller file (x264 CRF 23, fast)",
	                 "-c:v libx264 -crf 23 -preset veryfast");
	quality->addItem("Fastest, GPU (NVENC)",
	                 "-c:v h264_nvenc -cq 21 -preset p4");
	quality->setCurrentIndex(1);
	encForm->addRow("Quality", quality);

	/* The same two switches as the image panel; see offline.h. */
	auto *harden = new QCheckBox("Harden covers against un-censoring");
	harden->setChecked(true);
	harden->setToolTip(
	    "Pixelate, blur and the other effects that sample the picture get "
	    "a random shift per block, fine grain and a faint BETACENSOR "
	    "watermark. The pattern is fixed for the whole file, so averaging "
	    "frames does not wash it out.");
	auto *badge = new QCheckBox("Add a “censored with BetaCensor” badge");
	badge->setChecked(true);
	encForm->addRow("", harden);
	encForm->addRow("", badge);
	encForm->addRow(new QLabel(
	    "<i>The frame size and frame rate always match the source. Audio is "
	    "copied from the original untouched.<br>Effect, classes, confidence "
	    "and caption come from betacensor.conf — set those up in the "
	    "main window.</i>"));
	root->addWidget(encBox);

	/* ---- run ---- */
	auto *runRow = new QHBoxLayout;
	auto *start  = new QPushButton("Censor video");
	auto *cancel = new QPushButton("Cancel");
	cancel->setEnabled(false);
	auto *bar = new QProgressBar;
	bar->setRange(0, 100);
	bar->setValue(0);
	runRow->addWidget(start);
	runRow->addWidget(cancel);
	runRow->addWidget(bar, 1);
	root->addLayout(runRow);

	auto *log = new QPlainTextEdit;
	log->setReadOnly(true);
	log->setMaximumBlockCount(2000);
	QFont mono("monospace");
	mono.setStyleHint(QFont::TypeWriter);
	log->setFont(mono);
	root->addWidget(log, 1);

	auto *proc = new QProcess(&win);
	proc->setProcessChannelMode(QProcess::MergedChannels);

	/* ---- behaviour ---- */

	auto probe = [info](const QString &path) {
		if (path.isEmpty() || !QFileInfo::exists(path)) {
			info->setText("—");
			return;
		}
		QProcess p;
		p.start("ffprobe", { "-v", "error", "-select_streams", "v:0",
		                     "-show_entries",
		                     "stream=width,height,r_frame_rate",
		                     "-of", "default=nw=1:nk=1", path });
		if (!p.waitForFinished(4000)) {
			info->setText("could not probe (is ffmpeg installed?)");
			return;
		}
		const QStringList f =
		    QString::fromUtf8(p.readAllStandardOutput())
		        .split('\n', Qt::SkipEmptyParts);
		if (f.size() < 3) {
			info->setText("no video stream found");
			return;
		}
		double fps = 0.0;
		const QStringList r = f[2].split('/');
		if (r.size() == 2 && r[1].toDouble() != 0.0)
			fps = r[0].toDouble() / r[1].toDouble();
		info->setText(QString("%1 × %2 at %3 fps")
		                  .arg(f[0].trimmed(), f[1].trimmed())
		                  .arg(fps, 0, 'f', 3));
	};

	QObject::connect(inBrowse, &QPushButton::clicked, [&win, inEdit] {
		const QString p = QFileDialog::getOpenFileName(
		    &win, "Video to censor", QString(),
		    "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv);;"
		    "All files (*)");
		if (!p.isEmpty())
			inEdit->setText(p);
	});
	QObject::connect(outBrowse, &QPushButton::clicked, [&win, outEdit] {
		const QString p = QFileDialog::getSaveFileName(
		    &win, "Save censored video as", outEdit->text(),
		    "Video files (*.mp4 *.mkv);;All files (*)");
		if (!p.isEmpty())
			outEdit->setText(p);
	});
	QObject::connect(inEdit, &QLineEdit::textChanged,
	                 [outEdit, probe](const QString &t) {
		probe(t);
		/* Only fill the output in if the user has not typed one. */
		if (outEdit->text().isEmpty() ||
		    outEdit->property("auto").toBool()) {
			outEdit->setProperty("auto", true);
			outEdit->setText(t.isEmpty() ? QString() : suggestOutput(t));
		}
	});
	QObject::connect(outEdit, &QLineEdit::textEdited, [outEdit] {
		outEdit->setProperty("auto", false);
	});

	QObject::connect(start, &QPushButton::clicked,
	                 [&win, proc, inEdit, outEdit, quality, harden, badge,
	                  log, bar, start, cancel] {
		const QString in  = inEdit->text().trimmed();
		const QString out = outEdit->text().trimmed();
		if (in.isEmpty() || out.isEmpty()) {
			QMessageBox::warning(&win, "BetaCensor",
			                     "Choose an input and an output file.");
			return;
		}
		if (QFileInfo(in).canonicalFilePath() ==
		    QFileInfo(out).canonicalFilePath() && !out.isEmpty()) {
			QMessageBox::warning(&win, "BetaCensor",
			    "The output must be a different file from the input — "
			    "this writes the output while it is still reading the "
			    "input.");
			return;
		}
		if (QFileInfo::exists(out) &&
		    QMessageBox::question(&win, "BetaCensor",
		        QString("%1 already exists. Overwrite it?")
		            .arg(QFileInfo(out).fileName())) != QMessageBox::Yes)
			return;

		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert("BC_VIDEO_ENC", quality->currentData().toString());
		proc->setProcessEnvironment(env);

		log->clear();
		bar->setValue(0);
		start->setEnabled(false);
		cancel->setEnabled(true);
		QStringList args;
		if (!harden->isChecked())
			args << "--no-harden";
		if (!badge->isChecked())
			args << "--no-badge";
		args << "--" << in << out;
		log->appendPlainText("$ bc-video " + args.join(' '));
		proc->start(videoBinary(), args);
	});

	QObject::connect(cancel, &QPushButton::clicked, [proc, log] {
		log->appendPlainText("-- cancelled --");
		/*
		 * terminate() first: bc-video closes its pipes on the way out,
		 * which lets the encoder finalise whatever it has already been
		 * given rather than leaving a truncated file with no index.
		 */
		proc->terminate();
		if (!proc->waitForFinished(3000))
			proc->kill();
	});

	QObject::connect(proc, &QProcess::readyRead, [proc, log, bar] {
		const QString chunk = QString::fromUtf8(proc->readAll());
		for (const QString &line :
		     chunk.split('\n', Qt::SkipEmptyParts)) {
			log->appendPlainText(line.trimmed());
			static const QRegularExpression re(
			    "(\\d+)/(\\d+) frames");
			const auto m = re.match(line);
			if (m.hasMatch()) {
				const double d = m.captured(2).toDouble();
				if (d > 0)
					bar->setValue(int(100.0 *
					    m.captured(1).toDouble() / d));
			}
		}
	});

	QObject::connect(proc, QOverload<int, QProcess::ExitStatus>::of(
	                           &QProcess::finished),
	                 [log, bar, start, cancel](int code,
	                                           QProcess::ExitStatus st) {
		start->setEnabled(true);
		cancel->setEnabled(false);
		if (st == QProcess::NormalExit && code == 0) {
			bar->setValue(100);
			log->appendPlainText("-- done --");
		} else {
			log->appendPlainText(
			    QString("-- failed (exit %1) --").arg(code));
		}
	});

	QObject::connect(proc, &QProcess::errorOccurred,
	                 [log, start, cancel](QProcess::ProcessError e) {
		start->setEnabled(true);
		cancel->setEnabled(false);
		if (e == QProcess::FailedToStart)
			log->appendPlainText(
			    "-- could not start bc-video: build it with 'make video' --");
	});

	if (argc > 1)
		inEdit->setText(QString::fromLocal8Bit(argv[1]));

	win.resize(720, 560);
	win.show();
	return app.exec();
}

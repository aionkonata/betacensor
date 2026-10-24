/*
 * bc-image-gui -- front end for the offline image pass.
 *
 * A sibling of bc-video-gui, and separate from the live settings panel for the
 * same reason: this starts a batch that runs to completion and produces files,
 * where the live panel edits settings that apply the moment they change.
 *
 * Unlike the video panel it asks which classes to cover. That is the point of
 * the feature -- pick the body parts, point it at a folder -- and a per-run
 * choice should not have to go through the live censor's config to get there.
 * The choice is passed as --classes and never written back: the boxes start
 * out ticked the way betacensor.conf has them, and unticking one here leaves
 * the live censor alone. Effect, strength, floors and sectioning still come
 * from the config, so what is drawn matches what the screen would draw.
 *
 * The preview window asks a long-running `bc-image --serve` for each image,
 * so what it shows is the batch's own output rather than a reconstruction.
 *
 * No Q_OBJECT, so there is no moc step: every connection is a lambda. The
 * subclasses only override virtuals, which needs no meta-object.
 */

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollArea>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include <QButtonGroup>
#include <QCollator>
#include <QDirIterator>
#include <QHash>
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QRadioButton>
#include <QRandomGenerator>
#include <QShortcut>
#include <QCloseEvent>
#include <QFile>
#include <QMouseEvent>
#include <QSaveFile>
#include <QSignalBlocker>

#include <cmath>
#include <QVector>

#include <algorithm>
#include <functional>

namespace {

/* What the folder walk in bc-image reads; kept in step with READ_EXT. */
const QStringList IMAGE_EXT = { "png", "jpg", "jpeg", "jfif", "webp", "bmp",
                                "gif", "avif", "tif", "tiff" };

/*
 * The inputs as a batch would see them: folders expanded recursively, hidden
 * entries skipped, and any folder carrying bc-image's output marker skipped
 * along with everything under it -- the same rules as walk() in image.c, so
 * the preview offers exactly the images a run would censor.
 */
QStringList expandInputs(const QStringList &inputs)
{
	QStringList out;
	QHash<QString, bool> isOutput;
	auto underOutput = [&isOutput](const QString &root, QString dir) {
		while (dir.size() > root.size()) {
			auto it = isOutput.find(dir);
			bool m;
			if (it == isOutput.end()) {
				m = QFileInfo::exists(dir + "/.betacensor-output");
				isOutput.insert(dir, m);
			} else {
				m = *it;
			}
			if (m)
				return true;
			dir = QFileInfo(dir).path();
		}
		return false;
	};
	QCollator natural;
	natural.setNumericMode(true);
	for (const QString &in : inputs) {
		QFileInfo fi(in);
		if (!fi.isDir()) {
			out << fi.absoluteFilePath();
			continue;
		}
		const QString root = fi.absoluteFilePath();
		QStringList found;
		QDirIterator it(root, QDir::Files, QDirIterator::Subdirectories);
		while (it.hasNext() && found.size() < 20000) {
			const QString p = it.next();
			if (!IMAGE_EXT.contains(QFileInfo(p).suffix().toLower()))
				continue;
			if (underOutput(root, QFileInfo(p).path()))
				continue;
			found << p;
		}
		std::sort(found.begin(), found.end(),
		          [&natural](const QString &a, const QString &b) {
			return natural.compare(a, b) < 0;
		});
		out << found;
	}
	out.removeDuplicates();
	return out;
}

/*
 * Calls `f` when the watched widget is closed. The preview is a window of its
 * own, so without this, closing the main window left it -- and the model it
 * holds on the GPU -- running, and the application with it.
 */
class CloseWatch : public QObject {
public:
	explicit CloseWatch(std::function<void()> f, QObject *parent)
	    : QObject(parent), f_(std::move(f)) {}

protected:
	bool eventFilter(QObject *, QEvent *e) override
	{
		if (e->type() == QEvent::Close)
			f_();
		return false;
	}

private:
	std::function<void()> f_;
};

/* ------------------------------------------------------------------ */

/* One answer from the preview server. */
struct Preview {
	QString path;
	int     w = 0, h = 0;
	bool    animated = false;
	/* Exactly where effects are drawn. An eye bar is turned to the eye
	 * line, so a cover is a rectangle and an angle (radians, GL's:
	 * counter-clockwise on screen). */
	struct Cover {
		QRect  r;
		double angle;
		bool   manual;          /* drawn by the user, not found */
	};
	QVector<Cover> covers;
	struct Det {
		QRect   r;
		int     state;          /* 1 covered, 0 found but not ticked,
		                         * 2 covered until the user removed it */
		double  score;
		QString name;
	};
	QVector<Det> dets;
	QImage  original, censored;
};

/*
 * A long-running `bc-image --serve`. The model stays loaded, so stepping
 * through images costs one inference each and ticking a class costs none --
 * the server keeps the last image's detections. Only the newest request is
 * kept while one is in flight: ticking five boxes quickly is one redraw, not
 * a queue of five.
 */
class PreviewServer {
public:
	std::function<void()>                 onReady = [] {};
	std::function<void(const Preview &)>  onResult = [](const Preview &) {};
	std::function<void(const QString &)>  onError = [](const QString &) {};

	explicit PreviewServer(QObject *parent, QString binary)
	    : binary_(std::move(binary)), proc_(new QProcess(parent))
	{
		QObject::connect(proc_, &QProcess::readyReadStandardOutput,
		                 [this] { readSome(); });
		QObject::connect(proc_, &QProcess::readyReadStandardError, [this] {
			stderrTail_ += proc_->readAllStandardError();
			stderrTail_ = stderrTail_.right(4000);
		});
		QObject::connect(proc_, QOverload<int, QProcess::ExitStatus>::of(
		                            &QProcess::finished),
		                 [this](int code, QProcess::ExitStatus st) {
			const bool wasClosing = closing_;
			ready_ = busy_ = false;
			buf_.clear();
			state_ = WaitReady;
			if (wasClosing)
				return;
			/* 3: the model changed in the settings. Start over
			 * with the new one and ask again. */
			if (st == QProcess::NormalExit && code == 3) {
				if (!pending_ && haveLast_) {
					pending_ = true;
					pendClasses_ = lastClasses_;
					pendPath_ = lastPath_;
				}
				start();
				return;
			}
			const QString tail = QString::fromUtf8(stderrTail_)
			    .trimmed().section('\n', -2);
			onError("The preview stopped (exit " + QString::number(code) +
			        "). " + tail);
		});
	}
	~PreviewServer() { stop(); }

	void start()
	{
		if (proc_->state() != QProcess::NotRunning)
			return;
		closing_ = false;
		stderrTail_.clear();
		QStringList args{ "--serve" };
		/* The same file the batch reads, so what is previewed is what is
		 * written. The server re-reads it when it changes. */
		if (!editsPath_.isEmpty())
			args << "--edits" << editsPath_;
		proc_->start(binary_, args);
	}
	void setEditsPath(const QString &p) { editsPath_ = p; }
	void stop()
	{
		if (proc_->state() == QProcess::NotRunning)
			return;
		closing_ = true;
		proc_->closeWriteChannel();      /* stdin EOF: it exits */
		if (!proc_->waitForFinished(2000))
			proc_->kill();
	}
	bool running() const { return proc_->state() != QProcess::NotRunning; }

	void request(const QString &classes, const QString &path)
	{
		pending_ = true;
		pendClasses_ = classes;
		pendPath_ = path;
		pump();
	}

private:
	enum State { WaitReady, Header, Lines, Pixels };

	void pump()
	{
		if (!ready_ || busy_ || !pending_)
			return;
		pending_ = false;
		busy_ = true;
		lastClasses_ = pendClasses_;
		lastPath_ = pendPath_;
		haveLast_ = true;
		cur_ = Preview();
		cur_.path = pendPath_;
		proc_->write((pendClasses_ + "\t" + pendPath_ + "\n").toUtf8());
		state_ = Header;
	}

	bool takeLine(QByteArray &line)
	{
		const int nl = buf_.indexOf('\n');
		if (nl < 0)
			return false;
		line = buf_.left(nl);
		buf_.remove(0, nl + 1);
		return true;
	}

	void readSome()
	{
		buf_ += proc_->readAllStandardOutput();
		for (;;) {
			QByteArray line;
			if (state_ == WaitReady) {
				if (!takeLine(line))
					return;
				if (line.startsWith("READY")) {
					ready_ = true;
					state_ = Header;
					onReady();
					pump();
				}
			} else if (state_ == Header) {
				if (!busy_ || !takeLine(line))
					return;
				const QList<QByteArray> f = line.split(' ');
				if (f.value(0) != "OK") {
					busy_ = false;
					onError(QString::fromUtf8(line.mid(4)));
					pump();
					continue;
				}
				cur_.w = f.value(1).toInt();
				cur_.h = f.value(2).toInt();
				linesLeft_ = f.value(3).toInt() + f.value(4).toInt();
				cur_.animated = f.value(5).toInt() != 0;
				state_ = linesLeft_ ? Lines : Pixels;
			} else if (state_ == Lines) {
				if (!takeLine(line))
					return;
				const QList<QByteArray> f = line.split(' ');
				const QRect r(f.value(1).toInt(), f.value(2).toInt(),
				              f.value(3).toInt(), f.value(4).toInt());
				if (f.value(0) == "B")
					cur_.covers << Preview::Cover{
					    r, f.value(5).toDouble(), f.value(6) == "1" };
				else if (f.value(0) == "D")
					cur_.dets << Preview::Det{
					    r, f.value(5).toInt(), f.value(6).toDouble(),
					    QString::fromUtf8(f.value(7)) };
				if (--linesLeft_ == 0)
					state_ = Pixels;
			} else {
				const qsizetype one = qsizetype(cur_.w) * cur_.h * 3;
				if (buf_.size() < 2 * one)
					return;
				auto img = [this](const char *p) {
					return QImage(reinterpret_cast<const uchar *>(p),
					              cur_.w, cur_.h, cur_.w * 3,
					              QImage::Format_RGB888).copy();
				};
				cur_.original = img(buf_.constData());
				cur_.censored = img(buf_.constData() + one);
				buf_.remove(0, 2 * one);
				busy_ = false;
				state_ = Header;
				/* A newer request supersedes this answer: skip
				 * drawing it and go straight to the next. */
				if (pending_)
					pump();
				else
					onResult(cur_);
			}
		}
	}

	QString    binary_, editsPath_;
	QProcess  *proc_;
	QByteArray buf_, stderrTail_;
	State      state_ = WaitReady;
	bool       ready_ = false, busy_ = false, closing_ = false;
	bool       pending_ = false, haveLast_ = false;
	QString    pendClasses_, pendPath_, lastClasses_, lastPath_;
	int        linesLeft_ = 0;
	Preview    cur_;
};

/* ------------------------------------------------------------------ */

/*
 * The image, fitted to the widget or at 1:1, and in the Detections view the
 * covers and detections drawn over the original. Covers are the real drawn
 * rectangles (after growth and joining); detections are the model's boxes
 * behind them, so the gap between the two is the dilation.
 */

class Canvas : public QWidget {
public:
	bool showDetections = false;
	bool actualSize = false;
	const Preview *pv = nullptr;
	/* A drag that drew a rectangle, and a click that did not (image px). */
	std::function<void(const QRect &)> onDrawRect = [](const QRect &) {};
	std::function<void(const QPoint &)> onClick = [](const QPoint &) {};

	Canvas() { setCursor(Qt::CrossCursor); }

	/* Image pixel -> widget pixel, from the last paint. For the debug hook,
	 * which drives the real mouse path. */
	QPointF toWidget(const QPointF &imagePx) const
	{
		return QPointF(dst_.x() + imagePx.x() * scale_,
		               dst_.y() + imagePx.y() * scale_);
	}

	QSize sizeHint() const override
	{
		return pv && actualSize ? QSize(pv->w, pv->h) : QSize(480, 360);
	}

protected:
	/* Widget pixel -> image pixel, from the last paint's placement. */
	QPoint toImage(const QPointF &w) const
	{
		return QPoint(int(std::floor((w.x() - dst_.x()) / scale_)),
		              int(std::floor((w.y() - dst_.y()) / scale_)));
	}

	void mousePressEvent(QMouseEvent *e) override
	{
		if (e->button() != Qt::LeftButton || !pv || pv->original.isNull())
			return;
		dragging_ = true;
		from_ = to_ = e->position();
		update();
	}
	void mouseMoveEvent(QMouseEvent *e) override
	{
		if (!dragging_)
			return;
		to_ = e->position();
		update();
	}
	void mouseReleaseEvent(QMouseEvent *e) override
	{
		if (!dragging_ || e->button() != Qt::LeftButton)
			return;
		dragging_ = false;
		to_ = e->position();
		update();
		if (!pv || scale_ <= 0.0)
			return;
		const QPointF d = to_ - from_;
		/* A few pixels of wobble is still a click. */
		if (std::abs(d.x()) < 5 && std::abs(d.y()) < 5) {
			onClick(toImage(from_));
			return;
		}
		QRect r = QRect(toImage(from_), toImage(to_)).normalized();
		r = r.intersected(QRect(0, 0, pv->w, pv->h));
		if (r.width() >= 4 && r.height() >= 4)
			onDrawRect(r);
	}

	void paintEvent(QPaintEvent *) override
	{
		QPainter p(this);
		p.fillRect(rect(), palette().window());
		if (!pv || pv->original.isNull())
			return;
		const QImage &img = showDetections ? pv->original : pv->censored;
		QRectF dst;
		double s;
		if (actualSize) {
			s = 1.0;
			dst = QRectF(0, 0, pv->w, pv->h);
		} else {
			s = std::min(width() / double(pv->w), height() / double(pv->h));
			const double w = pv->w * s, h = pv->h * s;
			dst = QRectF((width() - w) / 2, (height() - h) / 2, w, h);
		}
		dst_ = dst;
		scale_ = s;
		p.setRenderHint(QPainter::SmoothPixmapTransform, s < 1.0);
		p.drawImage(dst, img);

		auto map = [&dst, s](const QRect &r) {
			return QRectF(dst.x() + r.x() * s, dst.y() + r.y() * s,
			              r.width() * s, r.height() * s);
		};
		p.setRenderHint(QPainter::Antialiasing);
		if (showDetections)
			paintDetections(p, map);

		/* The rectangle being drawn, over either view. */
		if (dragging_) {
			p.setBrush(QColor(0, 190, 255, 50));
			p.setPen(QPen(QColor(0, 190, 255), 2, Qt::DashLine));
			p.drawRect(QRectF(from_, to_).normalized());
		}
	}

private:
	template <typename Map>
	void paintDetections(QPainter &p, Map map)
	{
		const QColor red(230, 30, 40), amber(255, 190, 0),
		             grey(150, 150, 150), green(0, 190, 90);
		/* Uncovered first, so a covered box is never drawn under one. */
		QFont f = font();
		f.setBold(true);
		p.setFont(f);
		for (const auto &d : pv->dets) {
			if (d.state == 1)
				continue;
			p.setPen(QPen(d.state == 2 ? grey : amber, 2, Qt::DashLine));
			p.setBrush(Qt::NoBrush);
			p.drawRect(map(d.r));
		}
		for (const auto &c : pv->covers) {
			const QColor col = c.manual ? green : red;
			p.setPen(QPen(col, 2));
			p.setBrush(QColor(col.red(), col.green(), col.blue(), 70));
			const QRectF m = map(c.r);
			if (c.angle == 0.0) {
				p.drawRect(m);
				continue;
			}
			/* Turned about its centre. Qt's y points down, so GL's
			 * counter-clockwise angle is a negative Qt rotation. */
			p.save();
			p.translate(m.center());
			p.rotate(-c.angle * 57.29577951308232);
			p.drawRect(QRectF(-m.width() / 2, -m.height() / 2, m.width(),
			                  m.height()));
			p.restore();
		}
		for (const auto &d : pv->dets) {
			if (d.state != 1)
				continue;
			p.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt::DotLine));
			p.setBrush(Qt::NoBrush);
			p.drawRect(map(d.r));
		}
		/* Labels last, on a backing so they read on any image. */
		const QFontMetrics fm(f);
		for (const auto &d : pv->dets) {
			QString t = QString("%1 %2").arg(d.name).arg(d.score, 0, 'f', 2);
			if (d.state == 2)
				t += " — removed";
			const QRectF r = map(d.r);
			QRectF lab(r.x(), r.y() - fm.height() - 2,
			           fm.horizontalAdvance(t) + 8, fm.height() + 2);
			if (lab.y() < dst_.y())
				lab.moveTop(r.y());
			p.setPen(Qt::NoPen);
			p.setBrush(d.state == 1 ? QColor(230, 30, 40, 220)
			         : d.state == 2 ? QColor(120, 120, 120, 220)
			                        : QColor(255, 190, 0, 220));
			p.drawRect(lab);
			p.setPen(d.state == 0 ? Qt::black : Qt::white);
			p.drawText(lab.adjusted(4, 0, 0, 0),
			           Qt::AlignVCenter | Qt::AlignLeft, t);
		}
		/* Hand-drawn covers carry no detection to label; say what they are. */
		for (const auto &c : pv->covers) {
			if (!c.manual)
				continue;
			const QRectF m = map(c.r);
			const QString t = "drawn by you — click to remove";
			QRectF lab(m.x(), m.y() - fm.height() - 2,
			           fm.horizontalAdvance(t) + 8, fm.height() + 2);
			if (lab.y() < dst_.y())
				lab.moveTop(m.y());
			p.setPen(Qt::NoPen);
			p.setBrush(QColor(0, 150, 70, 230));
			p.drawRect(lab);
			p.setPen(Qt::white);
			p.drawText(lab.adjusted(4, 0, 0, 0),
			           Qt::AlignVCenter | Qt::AlignLeft, t);
		}
	}

	QRectF  dst_;
	double  scale_ = 1.0;
	bool    dragging_ = false;
	QPointF from_, to_;
};

/* ------------------------------------------------------------------ */

/*
 * The edits made by hand in the preview, kept between sessions in a file the
 * batch reads too (the format is documented at edits_load() in image.c).
 * Per image, keyed by canonical path and the image's pixel size: a different
 * file under the same name would otherwise inherit coordinates that mean
 * nothing on it.
 */
class EditStore {
public:
	struct Op {
		bool  add;       /* a drawn cover; else a removal point */
		QRect r;         /* the cover; for a removal, x and y are the point */
	};
	struct Img {
		int         w = 0, h = 0;
		QVector<Op> ops;
	};

	explicit EditStore(QString file) : file_(std::move(file)) { load(); }
	const QString &file() const { return file_; }

	static QString keyOf(const QString &path)
	{
		const QString c = QFileInfo(path).canonicalFilePath();
		return c.isEmpty() ? QFileInfo(path).absoluteFilePath() : c;
	}
	/* The tab-separated file cannot carry a name with a tab or newline. */
	static bool storable(const QString &key)
	{
		return !key.contains('\t') && !key.contains('\n');
	}

	/* This image's edits, started afresh if its size is not the one they
	 * were made at. */
	Img &at(const QString &path, int w, int h)
	{
		Img &i = imgs_[keyOf(path)];
		if (i.w != w || i.h != h) {
			i.ops.clear();
			i.w = w;
			i.h = h;
		}
		return i;
	}
	const Img *find(const QString &path) const
	{
		auto it = imgs_.constFind(keyOf(path));
		return it == imgs_.constEnd() ? nullptr : &*it;
	}
	int editCount(const QString &path) const
	{
		const Img *i = find(path);
		return i ? int(i->ops.size()) : 0;
	}
	int imagesEdited() const
	{
		int n = 0;
		for (auto it = imgs_.constBegin(); it != imgs_.constEnd(); ++it)
			n += !it->ops.isEmpty();
		return n;
	}

	void addRect(const QString &path, int w, int h, const QRect &r)
	{
		at(path, w, h).ops.append({ true, r });
		save();
	}
	void addDrop(const QString &path, int w, int h, const QPoint &pt)
	{
		at(path, w, h).ops.append({ false, QRect(pt.x(), pt.y(), 0, 0) });
		save();
	}
	/* Take back a drawn cover (by its rectangle). */
	bool removeRect(const QString &path, int w, int h, const QRect &r)
	{
		Img &i = at(path, w, h);
		for (int k = int(i.ops.size()) - 1; k >= 0; k--)
			if (i.ops[k].add && i.ops[k].r == r) {
				i.ops.remove(k);
				save();
				return true;
			}
		return false;
	}
	/* Bring back whatever a click inside `area` removed. */
	int removeDropsIn(const QString &path, int w, int h, const QRect &area)
	{
		Img &i = at(path, w, h);
		int n = 0;
		for (int k = int(i.ops.size()) - 1; k >= 0; k--)
			if (!i.ops[k].add && area.contains(i.ops[k].r.topLeft())) {
				i.ops.remove(k);
				n++;
			}
		if (n)
			save();
		return n;
	}
	void undoLast(const QString &path, int w, int h)
	{
		Img &i = at(path, w, h);
		if (!i.ops.isEmpty()) {
			i.ops.removeLast();
			save();
		}
	}
	void clear(const QString &path, int w, int h)
	{
		at(path, w, h).ops.clear();
		save();
	}

private:
	void load()
	{
		QFile f(file_);
		if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
			return;
		while (!f.atEnd()) {
			const QStringList c = QString::fromUtf8(f.readLine())
			    .trimmed().split('\t');
			if (c.size() < 4)
				continue;
			const QStringList wh = c[1].split('x');
			if (wh.size() != 2)
				continue;
			Img &i = imgs_[c[0]];
			i.w = wh[0].toInt();
			i.h = wh[1].toInt();
			if (c[2] == "add" && c.size() >= 7)
				i.ops.append({ true, QRect(c[3].toInt(), c[4].toInt(),
				                           c[5].toInt(), c[6].toInt()) });
			else if (c[2] == "drop" && c.size() >= 5)
				i.ops.append({ false, QRect(c[3].toInt(), c[4].toInt(), 0, 0) });
		}
	}
	/* Whole file, atomically: bc-image may read it between any two writes. */
	void save()
	{
		QSaveFile f(file_);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Text))
			return;
		for (auto it = imgs_.constBegin(); it != imgs_.constEnd(); ++it) {
			if (!storable(it.key()))
				continue;
			for (const Op &o : it->ops) {
				QString line = QString("%1\t%2x%3\t").arg(it.key())
				                   .arg(it->w).arg(it->h);
				line += o.add
				    ? QString("add\t%1\t%2\t%3\t%4").arg(o.r.x()).arg(o.r.y())
				          .arg(o.r.width()).arg(o.r.height())
				    : QString("drop\t%1\t%2").arg(o.r.x()).arg(o.r.y());
				f.write(line.toUtf8() + '\n');
			}
		}
		f.commit();
	}

	QString               file_;
	QHash<QString, Img>   imgs_;
};

/* ------------------------------------------------------------------ */

/*
 * The preview window. It owns its server, started on first show and
 * stopped when the window closes, so a closed preview holds no GPU memory.
 */
class PreviewWindow : public QWidget {
public:
	/* The main window's live state, asked for on every refresh. */
	std::function<QString()>     classes = [] { return QString("-"); };
	/* "h" harden, "b" badge, "-" neither -- see serve() in image.c. */
	std::function<QString()>     flags = [] { return QString("hb"); };
	std::function<QStringList()> inputs = [] { return QStringList(); };

	explicit PreviewWindow(QWidget *owner, const QString &binary)
	    : QWidget(owner, Qt::Window), server_(this, binary)
	{
		setWindowTitle("BetaCensor — preview");
		auto *root = new QVBoxLayout(this);

		auto *top = new QHBoxLayout;
		pick_ = new QComboBox;
		pick_->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
		pick_->setMinimumContentsLength(30);
		auto *prev = new QPushButton("◀");
		auto *next = new QPushButton("▶");
		prev->setToolTip("Previous image (←)");
		next->setToolTip("Next image (→)");
		count_ = new QLabel;
		top->addWidget(pick_, 1);
		top->addWidget(prev);
		top->addWidget(next);
		top->addWidget(count_);
		root->addLayout(top);

		auto *modes = new QHBoxLayout;
		auto *censored = viewCensored_ = new QRadioButton("Censored");
		auto *dets = viewDets_ = new QRadioButton("Detections");
		censored->setToolTip("Exactly what the batch will write");
		dets->setToolTip("The original, with what will be covered in red "
		                 "and what was found but is not ticked in amber");
		censored->setChecked(true);
		auto *group = new QButtonGroup(this);
		group->addButton(censored, 0);
		group->addButton(dets, 1);
		auto *actual = new QCheckBox("Actual size");
		modes->addWidget(censored);
		modes->addWidget(dets);
		modes->addSpacing(16);
		modes->addWidget(actual);
		modes->addStretch(1);
		modes->addWidget(new QLabel("<i>Space switches view</i>"));
		root->addLayout(modes);

		canvas_ = new Canvas;
		scroll_ = new QScrollArea;
		scroll_->setWidget(canvas_);
		scroll_->setWidgetResizable(true);
		scroll_->setAlignment(Qt::AlignCenter);
		root->addWidget(scroll_, 1);

		/*
		 * Hand edits. The detector misses things and sometimes finds what is
		 * not there; both are fixed here, per image, and the batch writes
		 * what is shown. Fail-closed is why drawing is the easy gesture and
		 * removing is an explicit click on the thing itself.
		 */
		auto *hint = new QLabel(
		    "<i>Drag on the image to cover something it missed · click a "
		    "cover to remove it · click a removed one to bring it back</i>");
		hint->setWordWrap(true);
		root->addWidget(hint);
		auto *editRow = new QHBoxLayout;
		undoBtn_ = new QPushButton("Undo last edit");
		clearBtn_ = new QPushButton("Clear edits on this image");
		editsLabel_ = new QLabel;
		editRow->addWidget(undoBtn_);
		editRow->addWidget(clearBtn_);
		editRow->addWidget(editsLabel_, 1);
		root->addLayout(editRow);
		QObject::connect(undoBtn_, &QPushButton::clicked, [this] {
			if (haveImage())
				store_->undoLast(pv_.path, pv_.w, pv_.h);
			edited();
		});
		QObject::connect(clearBtn_, &QPushButton::clicked, [this] {
			if (haveImage())
				store_->clear(pv_.path, pv_.w, pv_.h);
			edited();
		});
		canvas_->onClick = [this](const QPoint &pt) { clickAt(pt); };
		canvas_->onDrawRect = [this](const QRect &r) { drawn(r); };

		status_ = new QLabel("Loading the model…");
		status_->setWordWrap(true);
		status_->setTextInteractionFlags(Qt::TextSelectableByMouse);
		root->addWidget(status_);
		auto *note = new QLabel(
		    "<i>These are the exact pixels the batch writes, before the "
		    "output format compresses them. Transparent areas are shown "
		    "on white.</i>");
		note->setWordWrap(true);
		root->addWidget(note);

		QObject::connect(pick_, QOverload<int>::of(&QComboBox::currentIndexChanged),
		                 [this](int) { refresh(); });
		QObject::connect(prev, &QPushButton::clicked, [this] { step(-1); });
		QObject::connect(next, &QPushButton::clicked, [this] { step(+1); });
		QObject::connect(group, &QButtonGroup::idClicked, [this](int id) {
			canvas_->showDetections = id == 1;
			canvas_->update();
		});
		QObject::connect(actual, &QCheckBox::toggled, [this](bool on) {
			canvas_->actualSize = on;
			scroll_->setWidgetResizable(!on);
			if (on && canvas_->pv)
				canvas_->resize(canvas_->pv->w, canvas_->pv->h);
			canvas_->update();
		});
		auto *sl = new QShortcut(QKeySequence(Qt::Key_Left), this);
		auto *sr = new QShortcut(QKeySequence(Qt::Key_Right), this);
		auto *sp = new QShortcut(QKeySequence(Qt::Key_Space), this);
		QObject::connect(sl, &QShortcut::activated, [this] { step(-1); });
		QObject::connect(sr, &QShortcut::activated, [this] { step(+1); });
		QObject::connect(sp, &QShortcut::activated, [dets, censored] {
			(dets->isChecked() ? censored : dets)->click();
		});

		server_.onReady = [this] {
			status_->setText("Ready.");
		};
		server_.onResult = [this](const Preview &pv) { show_(pv); };
		server_.onError = [this](const QString &m) {
			status_->setText("<b>" + m.toHtmlEscaped() + "</b>");
		};

		debounce_.setSingleShot(true);
		debounce_.setInterval(120);
		QObject::connect(&debounce_, &QTimer::timeout, [this] { send(); });

		resize(900, 760);
	}

	/* Open at `path` (a file, or the first image of a folder). */
	void openAt(const QString &path)
	{
		reloadInputs();
		const QString abs = QFileInfo(path).absoluteFilePath();
		int at = files_.indexOf(abs);
		if (at < 0)
			for (int i = 0; i < files_.size(); i++)
				if (files_[i].startsWith(abs + "/")) {
					at = i;
					break;
				}
		if (at >= 0)
			pick_->setCurrentIndex(at);
		show();
		raise();
		activateWindow();
		server_.start();
		refresh();
	}

	/* The main window's inputs changed: re-list, keeping our place. */
	void reloadInputs()
	{
		const QString keep = pick_->currentData().toString();
		files_ = expandInputs(inputs());
		QSignalBlocker block(pick_);
		pick_->clear();
		for (const QString &f : files_) {
			QFileInfo fi(f);
			pick_->addItem(fi.dir().dirName() + "/" + fi.fileName(), f);
		}
		const int at = files_.indexOf(keep);
		pick_->setCurrentIndex(at >= 0 ? at : 0);
		updateCount();
	}

	/* Something that changes the result changed: redraw, soon. */
	void refresh()
	{
		if (isVisible())
			debounce_.start();
	}

	void setDetectionsView(bool on)
	{
		(on ? viewDets_ : viewCensored_)->click();
	}
	/* The edits this window makes and its server reads. Set before the
	 * first show: the server is started with the store's file. */
	void setStore(EditStore *s)
	{
		store_ = s;
		server_.setEditsPath(s ? s->file() : QString());
	}
	/*
	 * DEBUG ONLY: a left-button press at `from` and release at `to` (image
	 * pixels), sent as real mouse events to the canvas -- so a test drives
	 * the same path a hand does, coordinate mapping included. A click is
	 * from == to.
	 */
	void debugMouse(const QPointF &from, const QPointF &to)
	{
		const QPointF a = canvas_->toWidget(from), b = canvas_->toWidget(to);
		auto ev = [this](QEvent::Type t, const QPointF &p) {
			QMouseEvent e(t, p, canvas_->mapToGlobal(p), Qt::LeftButton,
			              t == QEvent::MouseButtonRelease ? Qt::NoButton
			                                              : Qt::LeftButton,
			              Qt::NoModifier);
			QApplication::sendEvent(canvas_, &e);
		};
		ev(QEvent::MouseButtonPress, a);
		ev(QEvent::MouseMove, b);
		ev(QEvent::MouseButtonRelease, b);
	}
	std::function<void()> shown = [] {};
	/* Edits changed (here, or Clear): the main window may want to say so. */
	std::function<void()> editsChanged = [] {};

protected:
	void closeEvent(QCloseEvent *e) override
	{
		server_.stop();
		QWidget::closeEvent(e);
	}

private:
	bool haveImage() const { return have_ && store_ && pv_.w > 0; }

	/* Is the image pixel (x, y) inside this cover, turned as it is? The
	 * shader's own test, as bc_off_box_has() makes it. */
	static bool coverHas(const Preview::Cover &c, const QPoint &p)
	{
		if (c.angle == 0.0)
			return c.r.contains(p);
		const double rx = p.x() + 0.5 - (c.r.x() + c.r.width() * 0.5);
		const double ry = -(p.y() + 0.5 - (c.r.y() + c.r.height() * 0.5));
		const double cs = std::cos(-c.angle), sn = std::sin(-c.angle);
		const double lx = rx * cs - ry * sn, ly = rx * sn + ry * cs;
		return std::abs(lx) <= c.r.width() * 0.5 &&
		       std::abs(ly) <= c.r.height() * 0.5;
	}

	void edited()
	{
		updateEditsUi();
		editsChanged();
		refresh();
	}

	void updateEditsUi()
	{
		const int n = haveImage() ? store_->editCount(pv_.path) : 0;
		undoBtn_->setEnabled(n > 0);
		clearBtn_->setEnabled(n > 0);
		editsLabel_->setText(n ? QString("<b>%1 hand edit%2 on this image</b>")
		                             .arg(n).arg(n == 1 ? "" : "s")
		                       : QString());
	}

	/* A click: remove the cover under it, or bring back a removed one. */
	void clickAt(const QPoint &pt)
	{
		if (!haveImage()) {
			status_->setText("Nothing loaded yet.");
			return;
		}
		if (!EditStore::storable(EditStore::keyOf(pv_.path))) {
			status_->setText("<b>This file's name has a tab or line break in "
			                 "it, so edits cannot be saved for it.</b>");
			return;
		}
		/* A cover drawn by hand is taken back; later ones sit on top. */
		for (int i = int(pv_.covers.size()) - 1; i >= 0; i--) {
			const auto &c = pv_.covers[i];
			if (c.manual && coverHas(c, pt)) {
				store_->removeRect(pv_.path, pv_.w, pv_.h, c.r);
				edited();
				return;
			}
		}
		/* A found cover is removed -- as the point, so it still holds if the
		 * classes or the effect change and the boxes move. */
		for (const auto &c : pv_.covers)
			if (!c.manual && coverHas(c, pt)) {
				store_->addDrop(pv_.path, pv_.w, pv_.h, pt);
				edited();
				return;
			}
		/* One you removed earlier comes back. */
		for (const auto &d : pv_.dets)
			if (d.state == 2) {
				const int m = std::max(6, std::max(d.r.width(), d.r.height()) / 6);
				const QRect a = d.r.adjusted(-m, -m, m, m);
				if (a.contains(pt) &&
				    store_->removeDropsIn(pv_.path, pv_.w, pv_.h, a)) {
					edited();
					return;
				}
			}
		status_->setText("Nothing to remove there. Drag on the image to draw "
		                 "a cover.");
	}

	void drawn(const QRect &r)
	{
		if (!haveImage())
			return;
		if (!EditStore::storable(EditStore::keyOf(pv_.path))) {
			status_->setText("<b>This file's name has a tab or line break in "
			                 "it, so edits cannot be saved for it.</b>");
			return;
		}
		store_->addRect(pv_.path, pv_.w, pv_.h, r);
		edited();
	}

	void step(int d)
	{
		if (files_.isEmpty())
			return;
		const int n = int(files_.size());
		pick_->setCurrentIndex((pick_->currentIndex() + d + n) % n);
	}
	void updateCount()
	{
		count_->setText(files_.isEmpty() ? "no images"
		    : QString("%1 / %2").arg(pick_->currentIndex() + 1)
		                        .arg(files_.size()));
	}
	void send()
	{
		updateCount();
		const QString path = pick_->currentData().toString();
		if (path.isEmpty()) {
			status_->setText("Add images or a folder in the main window.");
			return;
		}
		if (path.contains('\n') || path.contains('\t')) {
			status_->setText("This file's name cannot be previewed (it "
			                 "contains a tab or line break).");
			return;
		}
		if (!server_.running())
			server_.start();
		status_->setText("Working on " + QFileInfo(path).fileName() + "…");
		/* The server's first two fields travel as one here. */
		server_.request(classes() + "\t" + flags(), path);
	}
	void show_(const Preview &pv)
	{
		pv_ = pv;
		have_ = true;
		canvas_->pv = &pv_;
		if (canvas_->actualSize)
			canvas_->resize(pv_.w, pv_.h);
		canvas_->update();

		QStringList cov, unc, gone;
		for (const auto &d : pv_.dets)
			(d.state == 1 ? cov : d.state == 2 ? gone : unc)
			    << QString("%1 %2").arg(d.name).arg(d.score, 0, 'f', 2);
		int drawnByHand = 0;
		for (const auto &c : pv_.covers)
			drawnByHand += c.manual;
		updateEditsUi();
		QString s = QString("<b>%1</b> — %2 × %3. ")
		                .arg(QFileInfo(pv_.path).fileName().toHtmlEscaped())
		                .arg(pv_.w).arg(pv_.h);
		s += pv_.covers.isEmpty()
		    ? QString("<b>Nothing will be covered.</b>")
		    : QString("<b>%1 area%2 covered</b>")
		          .arg(pv_.covers.size())
		          .arg(pv_.covers.size() == 1 ? "" : "s");
		if (!cov.isEmpty())
			s += ": " + cov.join(", ");
		if (drawnByHand)
			s += QString("<br><span style='color:#007a3a'>%1 drawn by you."
			             "</span>").arg(drawnByHand);
		if (!gone.isEmpty())
			s += "<br><span style='color:#777'>Removed by you: " +
			     gone.join(", ") + "</span>";
		if (!unc.isEmpty())
			s += "<br><span style='color:#b07800'>Found but not ticked: " +
			     unc.join(", ") + "</span>";
		if (pv_.animated)
			s += "<br><i>Animated: the whole animation is censored, "
			     "with tracking between frames. This shows its first "
			     "frame; edits you make here apply to every frame.</i>";
		status_->setText(s);
		shown();
	}

	PreviewServer server_;
	QComboBox    *pick_;
	QLabel       *count_, *status_;
	Canvas       *canvas_;
	QScrollArea  *scroll_;
	QStringList   files_;
	QTimer        debounce_;
	Preview       pv_;
	bool          have_ = false;
	QRadioButton *viewCensored_ = nullptr, *viewDets_ = nullptr;
	EditStore    *store_ = nullptr;
	QPushButton  *undoBtn_ = nullptr, *clearBtn_ = nullptr;
	QLabel       *editsLabel_ = nullptr;
};

QString imageBinary()
{
	const QString beside =
	    QCoreApplication::applicationDirPath() + "/bc-image";
	if (QFileInfo::exists(beside))
		return beside;
	return "build/bc-image";
}

/* Files and folders dropped from a file manager land in the list. */
class DropList : public QListWidget {
public:
	DropList()
	{
		setAcceptDrops(true);
		setSelectionMode(QAbstractItemView::ExtendedSelection);
	}
	void addPath(const QString &p)
	{
		const QString abs = QFileInfo(p).absoluteFilePath();
		for (int i = 0; i < count(); i++)
			if (item(i)->data(Qt::UserRole).toString() == abs)
				return;
		auto *it = new QListWidgetItem(
		    (QFileInfo(abs).isDir() ? "📁 " : "") + abs);
		it->setData(Qt::UserRole, abs);
		addItem(it);
	}
	QStringList paths() const
	{
		QStringList out;
		for (int i = 0; i < count(); i++)
			out << item(i)->data(Qt::UserRole).toString();
		return out;
	}
	std::function<void()> changed = [] {};

protected:
	void dragEnterEvent(QDragEnterEvent *e) override
	{
		if (e->mimeData()->hasUrls())
			e->acceptProposedAction();
	}
	void dragMoveEvent(QDragMoveEvent *e) override
	{
		e->acceptProposedAction();
	}
	void dropEvent(QDropEvent *e) override
	{
		for (const QUrl &u : e->mimeData()->urls())
			if (u.isLocalFile())
				addPath(u.toLocalFile());
		e->acceptProposedAction();
		changed();
	}
};

/*
 * Somewhere sensible to put the results: "<folder>-censored" beside a single
 * input folder, or "censored" beside loose files. Never inside the input --
 * bc-image would skip it, but a folder of copies sitting among the originals
 * invites the wrong one being shared.
 */
QString suggestOutput(const QStringList &in)
{
	if (in.isEmpty())
		return QString();
	QFileInfo fi(in.first());
	if (fi.isDir())
		return fi.absolutePath() + "/" + fi.fileName() + "-censored";
	return fi.absolutePath() + "/censored";
}

}  // namespace

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	/*
	 * One hardening seed for the session, handed to every bc-image this
	 * panel starts -- the preview's server and each batch -- through the
	 * environment, so the pattern the preview shows is the pattern the
	 * batch writes. Random and never written anywhere; a seed set by the
	 * caller is respected.
	 */
	if (qEnvironmentVariableIsEmpty("BC_HARDEN_SEED"))
		qputenv("BC_HARDEN_SEED",
		        QByteArray::number(QRandomGenerator::system()->generate64(),
		                           16));

	QWidget win;
	win.setWindowTitle("BetaCensor — images");
	auto *root = new QVBoxLayout(&win);

	/* ---- inputs ---- */
	auto *inBox = new QGroupBox("Images");
	auto *inLay = new QVBoxLayout(inBox);
	auto *list  = new DropList;
	list->setMinimumHeight(110);
	inLay->addWidget(list, 1);
	auto *inBtns = new QHBoxLayout;
	auto *addFiles  = new QPushButton("Add images…");
	auto *addFolder = new QPushButton("Add folder…");
	auto *remove    = new QPushButton("Remove");
	auto *clear     = new QPushButton("Clear");
	inBtns->addWidget(addFiles);
	inBtns->addWidget(addFolder);
	inBtns->addStretch(1);
	inBtns->addWidget(remove);
	inBtns->addWidget(clear);
	inLay->addLayout(inBtns);
	inLay->addWidget(new QLabel(
	    "<i>Folders are searched through, subfolders included, and their "
	    "layout is copied to the output. Drop files or folders here.</i>"));
	root->addWidget(inBox, 1);

	/* ---- output ---- */
	auto *outBox  = new QGroupBox("Output");
	auto *outForm = new QFormLayout(outBox);
	auto *outEdit = new QLineEdit;
	auto *outBrowse = new QPushButton("Browse…");
	auto *outRow  = new QHBoxLayout;
	outRow->addWidget(outEdit, 1);
	outRow->addWidget(outBrowse);
	outForm->addRow("Folder", outRow);

	auto *format = new QComboBox;
	format->addItem("Same as the source (GIF becomes PNG)", "same");
	format->addItem("PNG (lossless)", "png");
	format->addItem("JPEG", "jpg");
	format->addItem("WebP", "webp");
	outForm->addRow("Format", format);

	auto *overwrite = new QCheckBox(
	    "Replace images already in the output folder");
	overwrite->setToolTip(
	    "Off: images already done are skipped, so a stopped batch carries "
	    "on where it left off.");
	outForm->addRow("", overwrite);

	auto *harden = new QCheckBox("Harden covers against un-censoring");
	harden->setChecked(true);
	harden->setToolTip(
	    "Pixelate, blur and the other effects that sample the image get a "
	    "random shift per block, fine grain and a faint BETACENSOR "
	    "watermark, so tools that reverse a mosaic have nothing exact to "
	    "work from. Fill, bars and captions hide everything already and "
	    "are left alone.");
	auto *badge = new QCheckBox("Add a “censored with BetaCensor” badge");
	badge->setChecked(true);
	badge->setToolTip("White text on a black box, bottom-right, sized to "
	                  "the image.");
	outForm->addRow("", harden);
	outForm->addRow("", badge);
	root->addWidget(outBox);

	/* ---- classes ---- */
	auto *clsBox = new QGroupBox("What to cover");
	auto *clsLay = new QVBoxLayout(clsBox);
	auto *clsStatus = new QLabel("Loading the model's classes…");
	clsStatus->setWordWrap(true);
	clsLay->addWidget(clsStatus);
	auto *clsHost = new QWidget;
	auto *clsGrid = new QGridLayout(clsHost);
	clsGrid->setContentsMargins(0, 0, 0, 0);
	auto *clsScroll = new QScrollArea;
	clsScroll->setWidget(clsHost);
	clsScroll->setWidgetResizable(true);
	clsScroll->setFrameShape(QFrame::NoFrame);
	clsScroll->setMinimumHeight(230);
	clsLay->addWidget(clsScroll, 1);
	auto *clsBtns = new QHBoxLayout;
	auto *selAll  = new QPushButton("All");
	auto *selNone = new QPushButton("None");
	auto *selExp  = new QPushButton("Exposed only");
	auto *selConf = new QPushButton("As in settings");
	selConf->setToolTip("The classes the live censor covers "
	                    "(betacensor.conf)");
	clsBtns->addWidget(selAll);
	clsBtns->addWidget(selNone);
	clsBtns->addWidget(selExp);
	clsBtns->addWidget(selConf);
	clsBtns->addStretch(1);
	clsLay->addLayout(clsBtns);
	clsLay->addWidget(new QLabel(
	    "<i>Effect, strength, confidence and sectioning come from "
	    "betacensor.conf — set those up in the main window.</i>"));
	root->addWidget(clsBox, 2);

	struct Cls {
		QString    name;
		bool       inConfig;
		QCheckBox *box;
	};
	auto *classes = new QList<Cls>;

	/* ---- run ---- */
	auto *runRow = new QHBoxLayout;
	auto *previewBtn = new QPushButton("Preview…");
	previewBtn->setToolTip("See exactly what will be covered, image by "
	                       "image, before running the batch. Double-click "
	                       "an input to open it there.");
	runRow->addWidget(previewBtn);
	auto *start  = new QPushButton("Censor images");
	auto *stop   = new QPushButton("Stop");
	auto *open   = new QPushButton("Open output folder");
	stop->setEnabled(false);
	auto *bar = new QProgressBar;
	bar->setRange(0, 100);
	bar->setValue(0);
	runRow->addWidget(start);
	runRow->addWidget(stop);
	runRow->addWidget(bar, 1);
	runRow->addWidget(open);
	root->addLayout(runRow);

	auto *log = new QPlainTextEdit;
	log->setReadOnly(true);
	log->setMaximumBlockCount(5000);
	QFont mono("monospace");
	mono.setStyleHint(QFont::TypeWriter);
	log->setFont(mono);
	log->setMinimumHeight(120);
	root->addWidget(log, 1);

	auto *proc = new QProcess(&win);
	proc->setProcessChannelMode(QProcess::MergedChannels);

	/*
	 * The preview follows this window: its image list is these inputs,
	 * and its classes are these boxes, ticked or not, as they stand --
	 * the same string Start would pass. Until the class list has loaded,
	 * "-" asks for the config's own selection, which is what the boxes
	 * will start out as anyway.
	 */
	auto *preview = new PreviewWindow(&win, imageBinary());
	win.installEventFilter(new CloseWatch([preview] { preview->close(); },
	                                      &win));
	/*
	 * Hand edits from the preview, kept in the project directory beside the
	 * config (the launcher runs the panel from there) and handed to every
	 * bc-image this panel starts, preview server and batch alike.
	 */
	const QString editsFile = qEnvironmentVariableIsEmpty("BC_IMAGE_EDITS")
	    ? QDir::current().absoluteFilePath("betacensor-image-edits.tsv")
	    : QString::fromLocal8Bit(qgetenv("BC_IMAGE_EDITS"));
	auto *edits = new EditStore(editsFile);
	preview->setStore(edits);
	preview->inputs = [list] { return list->paths(); };
	preview->flags = [harden, badge] {
		QString f;
		if (harden->isChecked())
			f += 'h';
		if (badge->isChecked())
			f += 'b';
		return f.isEmpty() ? QString("-") : f;
	};
	QObject::connect(harden, &QCheckBox::toggled,
	                 [preview] { preview->refresh(); });
	QObject::connect(badge, &QCheckBox::toggled,
	                 [preview] { preview->refresh(); });
	preview->classes = [classes] {
		if (classes->isEmpty())
			return QString("-");
		QStringList picked;
		for (const auto &c : *classes)
			if (c.box->isChecked())
				picked << c.name;
		return picked.join(',');
	};
	QObject::connect(previewBtn, &QPushButton::clicked, [list, preview] {
		const auto sel = list->selectedItems();
		const QStringList all = list->paths();
		preview->openAt(!sel.isEmpty()
		    ? sel.first()->data(Qt::UserRole).toString()
		    : all.value(0));
	});
	QObject::connect(list, &QListWidget::itemDoubleClicked,
	                 [preview](QListWidgetItem *it) {
		preview->openAt(it->data(Qt::UserRole).toString());
	});

	/* ---- behaviour ---- */

	auto refreshOutput = [list, outEdit, preview] {
		if (outEdit->text().isEmpty() ||
		    outEdit->property("auto").toBool()) {
			outEdit->setProperty("auto", true);
			outEdit->setText(suggestOutput(list->paths()));
		}
		if (preview->isVisible()) {
			preview->reloadInputs();
			preview->refresh();
		}
	};
	list->changed = refreshOutput;
	QObject::connect(outEdit, &QLineEdit::textEdited, [outEdit] {
		outEdit->setProperty("auto", false);
	});

	QObject::connect(addFiles, &QPushButton::clicked,
	                 [&win, list, refreshOutput] {
		const QStringList ps = QFileDialog::getOpenFileNames(
		    &win, "Images to censor", QString(),
		    "Images (*.png *.jpg *.jpeg *.jfif *.webp *.bmp *.gif "
		    "*.avif *.tif *.tiff);;All files (*)");
		for (const QString &p : ps)
			list->addPath(p);
		refreshOutput();
	});
	QObject::connect(addFolder, &QPushButton::clicked,
	                 [&win, list, refreshOutput] {
		const QString p = QFileDialog::getExistingDirectory(
		    &win, "Folder of images to censor");
		if (!p.isEmpty())
			list->addPath(p);
		refreshOutput();
	});
	QObject::connect(remove, &QPushButton::clicked, [list, refreshOutput] {
		qDeleteAll(list->selectedItems());
		refreshOutput();
	});
	QObject::connect(clear, &QPushButton::clicked, [list, refreshOutput] {
		list->clear();
		refreshOutput();
	});
	QObject::connect(outBrowse, &QPushButton::clicked, [&win, outEdit] {
		const QString p = QFileDialog::getExistingDirectory(
		    &win, "Put the censored images in", outEdit->text());
		if (!p.isEmpty()) {
			outEdit->setProperty("auto", false);
			outEdit->setText(p);
		}
	});
	QObject::connect(open, &QPushButton::clicked, [outEdit] {
		const QString p = outEdit->text().trimmed();
		if (!p.isEmpty() && QFileInfo(p).isDir())
			QDesktopServices::openUrl(QUrl::fromLocalFile(p));
	});

	QObject::connect(selAll, &QPushButton::clicked, [classes] {
		for (auto &c : *classes)
			c.box->setChecked(true);
	});
	QObject::connect(selNone, &QPushButton::clicked, [classes] {
		for (auto &c : *classes)
			c.box->setChecked(false);
	});
	/* By name, since that is all a model tells us: every taxonomy here
	 * spells the uncovered variant with EXPOSED. */
	QObject::connect(selExp, &QPushButton::clicked, [classes] {
		for (auto &c : *classes)
			c.box->setChecked(
			    c.name.contains("EXPOSED", Qt::CaseInsensitive));
	});
	QObject::connect(selConf, &QPushButton::clicked, [classes] {
		for (auto &c : *classes)
			c.box->setChecked(c.inConfig);
	});

	/*
	 * The class list is the model's own, so it is asked for rather than
	 * hardcoded: models disagree on both the names and the order. Loading
	 * the model takes a second or two, so it runs in the background and
	 * Start waits for it.
	 */
	auto *lister = new QProcess(&win);
	QObject::connect(lister, QOverload<int, QProcess::ExitStatus>::of(
	                             &QProcess::finished),
	                 [lister, classes, clsGrid, clsStatus, start,
	                  preview](int code, QProcess::ExitStatus st) {
		const QStringList lines =
		    QString::fromUtf8(lister->readAllStandardOutput())
		        .split('\n', Qt::SkipEmptyParts);
		for (const QString &l : lines) {
			const QStringList f = l.split('\t');
			if (f.size() < 3)
				continue;
			Cls c{ f[1], f[2].trimmed() == "1", new QCheckBox(f[1]) };
			c.box->setChecked(c.inConfig);
			/* Every tick redraws the preview; the server keeps
			 * the detections, so this costs a redraw, not an
			 * inference. */
			QObject::connect(c.box, &QCheckBox::toggled,
			                 [preview] { preview->refresh(); });
			const int i = int(classes->size());
			clsGrid->addWidget(c.box, i / 2, i % 2);
			classes->append(c);
		}
		if (st != QProcess::NormalExit || code != 0 || classes->isEmpty()) {
			const QString err = QString::fromUtf8(
			    lister->readAllStandardError()).trimmed();
			clsStatus->setText(
			    "Could not read the model's classes. Check the model "
			    "in the main settings window.\n" +
			    err.section('\n', -2));
			return;
		}
		clsStatus->setText(QString("%1 classes. Ticked ones are covered; "
		                           "this choice is for this run only.")
		                       .arg(classes->size()));
		start->setEnabled(true);
	});
	start->setEnabled(false);
	lister->start(imageBinary(), { "--list-classes" });

	QObject::connect(start, &QPushButton::clicked,
	                 [&win, proc, list, outEdit, format, overwrite, classes,
	                  harden, badge, edits, log, bar, start, stop] {
		const QStringList in = list->paths();
		const QString out = outEdit->text().trimmed();
		if (in.isEmpty() || out.isEmpty()) {
			QMessageBox::warning(&win, "BetaCensor",
			    "Add some images or a folder, and choose where the "
			    "censored copies go.");
			return;
		}
		QStringList picked;
		for (const auto &c : *classes)
			if (c.box->isChecked())
				picked << c.name;
		if (picked.isEmpty()) {
			QMessageBox::warning(&win, "BetaCensor",
			    "Tick at least one thing to cover — with nothing "
			    "ticked the copies would be identical to the "
			    "originals.");
			return;
		}
		const QString outAbs = QFileInfo(out).absoluteFilePath();
		for (const QString &p : in)
			if (QFileInfo(p).isDir() &&
			    QFileInfo(p).canonicalFilePath() ==
			        QFileInfo(outAbs).canonicalFilePath()) {
				QMessageBox::warning(&win, "BetaCensor",
				    "The output folder must not be one of the "
				    "input folders — the copies would replace "
				    "the originals.");
				return;
			}

		QStringList args{ "--classes", picked.join(','),
		                  "--format", format->currentData().toString(),
		                  "--out", outAbs };
		if (overwrite->isChecked())
			args << "--overwrite";
		if (!harden->isChecked())
			args << "--no-harden";
		if (!badge->isChecked())
			args << "--no-badge";
		/* What was drawn and removed by hand in the preview. Passed whenever
		 * there are any: they only touch the images they were made on. */
		const int edited = edits->imagesEdited();
		if (edited)
			args << "--edits" << edits->file();
		/* "--" so an input whose name starts with '-' is not read as
		 * an option. */
		args << "--" << in;

		log->clear();
		bar->setValue(0);
		start->setEnabled(false);
		stop->setEnabled(true);
		if (edited)
			log->appendPlainText(
			    QString("-- applying your hand edits on %1 image%2 --")
			        .arg(edited).arg(edited == 1 ? "" : "s"));
		log->appendPlainText(
		    QString("$ bc-image --classes <%1 of %2> %3 -- <%4 input%5>")
		        .arg(picked.size()).arg(classes->size())
		        .arg(args.mid(2, args.indexOf("--") - 2).join(' '))
		        .arg(in.size()).arg(in.size() == 1 ? "" : "s"));
		proc->start(imageBinary(), args);
	});

	QObject::connect(stop, &QPushButton::clicked, [proc, log, stop] {
		/*
		 * SIGTERM, not kill: bc-image finishes the image it is on and
		 * stops between images, so no half-written file is left and
		 * the next run resumes from the first image not done.
		 */
		log->appendPlainText("-- stopping after the current image --");
		stop->setEnabled(false);
		proc->terminate();
	});

	QObject::connect(proc, &QProcess::readyRead, [proc, log, bar] {
		const QString chunk = QString::fromUtf8(proc->readAll());
		for (const QString &line :
		     chunk.split('\n', Qt::SkipEmptyParts)) {
			const QString t = line.trimmed();
			/* The detector's own start-up chatter is noise here. */
			if (t.contains("onnxruntime") || t.contains("sections:"))
				continue;
			log->appendPlainText(t);
			static const QRegularExpression re("\\[(\\d+)/(\\d+)\\]");
			const auto m = re.match(t);
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
	                 [log, bar, start, stop](int code,
	                                         QProcess::ExitStatus st) {
		start->setEnabled(true);
		stop->setEnabled(false);
		if (st == QProcess::NormalExit && code == 0) {
			bar->setValue(100);
			log->appendPlainText("-- done --");
		} else {
			log->appendPlainText(QString(
			    "-- not complete (exit %1): see above. Run again to "
			    "carry on; finished images are skipped --").arg(code));
		}
	});

	QObject::connect(proc, &QProcess::errorOccurred,
	                 [log, start, stop](QProcess::ProcessError e) {
		start->setEnabled(true);
		stop->setEnabled(false);
		if (e == QProcess::FailedToStart)
			log->appendPlainText(
			    "-- could not start bc-image: build it with 'make image' --");
	});

	for (int i = 1; i < argc; i++)
		list->addPath(QString::fromLocal8Bit(argv[i]));
	refreshOutput();

	win.resize(760, 860);
	win.show();

	/* BC_GUI_SHOT=<path>: render the window once the classes are in, and
	 * exit -- see the same hook in gui.cpp for why it exists. */
	const QByteArray shot = qgetenv("BC_GUI_SHOT");
	if (!shot.isEmpty()) {
		const QByteArray sz = qgetenv("BC_GUI_SHOT_SIZE");
		if (!sz.isEmpty()) {
			const QList<QByteArray> wh = sz.split('x');
			if (wh.size() == 2)
				win.resize(wh[0].toInt(), wh[1].toInt());
		}
		const QString path = QString::fromLocal8Bit(shot);
		/*
		 * BC_IMAGE_GUI_AUTORUN=1 also presses Start once the classes
		 * are in, and takes the shot when the batch ends: the whole
		 * panel, argument assembly included, driven without a human.
		 * DEBUG ONLY.
		 */
		/*
		 * BC_IMAGE_GUI_PREVIEW=censored|detections opens the preview
		 * on the first input once the classes are in, and shoots the
		 * preview window instead, after its first image arrives.
		 */
		const QByteArray pmode = qgetenv("BC_IMAGE_GUI_PREVIEW");
		if (!pmode.isEmpty()) {
			QObject::connect(lister,
			    QOverload<int, QProcess::ExitStatus>::of(
			        &QProcess::finished),
			    [preview, list, pmode] {
				preview->setDetectionsView(pmode == "detections");
				preview->openAt(list->paths().value(0));
			});
			/* ...and BC_IMAGE_GUI_PREVIEW_UNTICK=NAME unticks that
			 * class after the first image, shooting the redraw. */
			const QString untick = QString::fromLocal8Bit(
			    qgetenv("BC_IMAGE_GUI_PREVIEW_UNTICK"));
			/*
			 * BC_IMAGE_GUI_PREVIEW_MOUSE="x0,y0,x1,y1;..." drives the real
			 * mouse path after the first image: each item is a press at
			 * (x0,y0) and a release at (x1,y1) in image pixels -- a click
			 * when they are equal, a drawn rectangle when not. One item is
			 * played per result, and the shot is taken after the last.
			 */
			const QStringList mouse = QString::fromLocal8Bit(
			    qgetenv("BC_IMAGE_GUI_PREVIEW_MOUSE"))
			        .split(';', Qt::SkipEmptyParts);
			auto *seen = new int(0);
			preview->shown = [preview, path, untick, classes, seen, mouse] {
				if (!untick.isEmpty() && (*seen)++ == 0) {
					for (auto &c : *classes)
						if (c.name == untick)
							c.box->setChecked(false);
					return;
				}
				if (!mouse.isEmpty() && *seen < mouse.size()) {
					const QStringList v = mouse[(*seen)++].split(',');
					if (v.size() == 4) {
						preview->debugMouse(
						    QPointF(v[0].toDouble(), v[1].toDouble()),
						    QPointF(v[2].toDouble(), v[3].toDouble()));
						return;
					}
				}
				QTimer::singleShot(300, [preview, path] {
					preview->grab().save(path);
					QApplication::quit();
				});
			};
			return app.exec();
		}
		const bool autorun = !qgetenv("BC_IMAGE_GUI_AUTORUN").isEmpty();
		QObject *done = autorun ? static_cast<QObject *>(proc) : lister;
		auto snap = [&win, path] {
			QTimer::singleShot(500, [&win, path] {
				win.grab().save(path);
				QApplication::quit();
			});
		};
		if (autorun)
			QObject::connect(lister,
			    QOverload<int, QProcess::ExitStatus>::of(
			        &QProcess::finished),
			    [start] { QTimer::singleShot(0, start, &QPushButton::click); });
		QObject::connect(static_cast<QProcess *>(done),
		    QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
		    snap);
	}
	return app.exec();
}

/*
 * betacensor-gui -- control panel for the running censor (DESIGN.md §7 phase 4).
 *
 * The config file *is* the IPC. This process never talks to the core directly:
 * it edits betacensor.conf, and the core notices the mtime move at 4 Hz. No
 * socket, no D-Bus, and the core keeps zero GUI dependencies.
 *
 * Two consequences worth stating, because they shape the whole file:
 *
 * [1] The class list is not hardcoded and this binary does not link ONNX. The
 *     taxonomy comes from whichever model the core loaded -- our two models
 *     disagree on both order and count -- so the core writes the class names it
 *     actually has into the file, and the GUI renders a checkbox for whatever it
 *     finds under [classes]. Switch models and the GUI follows with no changes.
 *
 * [2] Saving rewrites lines in place rather than regenerating the file. That
 *     preserves comments, ordering, and any key this build does not know about,
 *     so an older GUI can never silently drop a newer core's settings.
 *
 * No Q_OBJECT anywhere, so there is no moc step: every connection is a lambda.
 */

#include <QApplication>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileSystemWatcher>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QMessageBox>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPalette>
#include <QPushButton>
#include <QGridLayout>
#include <QResizeEvent>
#include <QHBoxLayout>
#include <QFont>
#include <QFontDatabase>
#include <QScrollArea>
#include <QSpinBox>
#include <QStyleFactory>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <cstdio>   /* ::rename -- see save() */
#include <QTextStream>
#include <QStringList>

#include <functional>

namespace {

const char *DEFAULT_CONFIG = "betacensor.conf";

/* --- look ------------------------------------------------------------------
 *
 * The panel ships its own dark theme instead of inheriting the desktop's.
 *
 * That reverses an earlier decision here, so the reason matters. This file used
 * to set no colours at all, on the grounds that a hardcoded colour is wrong
 * under a theme that did not expect it -- correct, as far as it went. What it
 * could not do is produce a *considered* layout: with every surface the same
 * flat palette role there is nothing to group rows into cards with, nothing to
 * make a heading read as a heading, and no accent to mark what is on. The panel
 * that resulted was one undifferentiated column, which is the complaint.
 *
 * Owning the palette outright is the honest version of that. Fusion is forced
 * underneath it because the native Plasma style draws its own frames and
 * padding that no stylesheet here can predict -- half-themed is worse than
 * either end. Nothing below reads a palette role, so there is no theme this can
 * come out illegible under.
 */
namespace col {
const char *BG      = "#0e0e10";   /* window */
const char *CARD    = "#17171a";   /* grouped rows */
const char *LINE    = "#26262b";   /* card borders and row separators */
const char *INPUT   = "#1e1e22";
const char *EDGE    = "#303036";   /* input borders */
const char *TEXT    = "#e6e6ea";
const char *MUTED   = "#9a9aa4";   /* descriptions, headings */
const char *ACCENT  = "#e03b3b";
}

/*
 * `family` is repeated here as well as in QApplication::setFont because a rule
 * that sets any font property makes Qt re-resolve the whole font for that
 * widget, and the family it falls back to is the platform default -- monospace
 * on this machine. Setting the weight on a row title was enough to put that one
 * label back into a terminal font while everything around it stayed
 * proportional.
 */
QString styleSheet(const QString &family)
{
	return QString(R"(
* { font-family: "%FAMILY%"; }
QWidget#root, QTabWidget::pane { background: %BG%; }
QScrollArea { background: transparent; border: none; }

QFrame#card {
    background: %CARD%;
    border: 1px solid %LINE%;
    border-radius: 10px;
}
QFrame#sep { background: %LINE%; border: none; }

QLabel#sectionHeader {
    color: %MUTED%;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#rowTitle { font-weight: 600; color: %TEXT%; }
QLabel#rowDesc  { color: %MUTED%; }
QLabel#note     { color: %MUTED%; }
QLabel#status   { color: %MUTED%; }
QLabel#wordmark { color: %TEXT%; font-weight: 800; letter-spacing: 4px; }
QLabel#wordmarkAccent { color: %ACCENT%; font-weight: 800; letter-spacing: 4px; }

QTabWidget::pane { border: none; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab {
    background: transparent;
    color: %MUTED%;
    padding: 8px 14px;
    margin-right: 2px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}
QTabBar::tab:selected { color: %TEXT%; border-bottom: 2px solid %ACCENT%; }
QTabBar::tab:hover:!selected { color: %TEXT%; }

QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {
    background: %INPUT%;
    border: 1px solid %EDGE%;
    border-radius: 6px;
    padding: 5px 8px;
    color: %TEXT%;
}
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
    border: 1px solid %ACCENT%;
}
QComboBox::drop-down { border: none; width: 20px; }
/* The step arrows are left to the style to draw. Giving them a background here
 * takes Qt off the native path for that subcontrol, and it then wants an
 * `image:` for the arrow itself -- which there is nowhere to load from without
 * a resource file. The result was a spin box with no arrows at all, i.e. one
 * that did not look like a spin box. */
QComboBox QAbstractItemView {
    background: %INPUT%;
    border: 1px solid %EDGE%;
    selection-background-color: %ACCENT%;
    outline: none;
}

QPushButton {
    background: #232329;
    border: 1px solid #33333a;
    border-radius: 6px;
    padding: 6px 16px;
    color: %TEXT%;
    font-weight: 600;
}
QPushButton:hover  { background: #2b2b32; }
QPushButton:pressed { background: #1b1b20; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: #33333a; border-radius: 5px; min-height: 32px;
}
QScrollBar::handle:vertical:hover { background: #45454e; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
)")
	.replace("%FAMILY%", family)
	.replace("%BG%", col::BG).replace("%CARD%", col::CARD)
	.replace("%LINE%", col::LINE).replace("%INPUT%", col::INPUT)
	.replace("%EDGE%", col::EDGE).replace("%TEXT%", col::TEXT)
	.replace("%MUTED%", col::MUTED).replace("%ACCENT%", col::ACCENT);
}

/*
 * A sliding switch, for the settings that are on or off.
 *
 * A QCheckBox already carries the whole of the behaviour -- tri-state aside, it
 * is this widget -- so this overrides nothing but the painting. Worth the forty
 * lines because a tick box states a fact ("this is ticked") while a switch
 * states a position, and every boolean in this panel is something the censor is
 * or is not currently doing. It also reads at a glance down a column of twenty
 * class toggles, which a 13-pixel tick does not.
 *
 * No Q_OBJECT: nothing here declares a signal or a slot, so there is still no
 * moc step and the build stays one compile.
 */
class Toggle : public QCheckBox {
public:
	explicit Toggle(QWidget *parent = nullptr) : QCheckBox(parent)
	{
		setCursor(Qt::PointingHandCursor);
	}
	QSize sizeHint() const override { return QSize(44, 24); }
	QSize minimumSizeHint() const override { return QSize(44, 24); }

protected:
	/*
	 * The whole switch is the target.
	 *
	 * QCheckBox does not hit-test against its own rect: it asks the style
	 * for SE_CheckBoxClickRect, which is the tick indicator plus the label
	 * text. This widget has no text and paints its own pill, so that rect
	 * was a small box at the left edge and most of the switch -- including
	 * the knob, the part that looks like the control -- did nothing when
	 * clicked. Nothing about it appeared disabled, so the panel simply
	 * seemed not to work.
	 */
	bool hitButton(const QPoint &pos) const override
	{
		return rect().contains(pos);
	}

	void paintEvent(QPaintEvent *) override
	{
		QPainter p(this);
		p.setRenderHint(QPainter::Antialiasing, true);

		const QRectF r(0, (height() - 24) / 2.0, 44, 24);
		const qreal  rad = r.height() / 2.0;

		p.setPen(Qt::NoPen);
		p.setBrush(isChecked() ? QColor(col::ACCENT) : QColor("#34343c"));
		p.drawRoundedRect(r, rad, rad);

		/* Dim the whole control when disabled rather than only the knob,
		 * or an off-but-unavailable switch reads as merely off. */
		if (!isEnabled()) {
			p.setBrush(QColor(0, 0, 0, 110));
			p.drawRoundedRect(r, rad, rad);
		}

		const qreal d = r.height() - 6.0;
		const qreal x = isChecked() ? r.right() - 3.0 - d : r.left() + 3.0;
		p.setBrush(QColor("#f4f4f6"));
		p.drawEllipse(QRectF(x, r.top() + 3.0, d, d));
	}
};

/*
 * The class toggles, in as many columns as the window can hold.
 *
 * Two fixed columns do not survive a narrow window: a grid cannot shrink below
 * the width of its widest row, so the page grows wider than the viewport and --
 * with horizontal scrolling off, which it must be for the wrapping descriptions
 * elsewhere to work -- the right-hand column is simply not drawn. Twenty
 * classes silently became ten, in the one panel where "I cannot see the toggle"
 * and "the toggle is off" look identical.
 *
 * So the column count is a function of the width instead, recomputed on resize.
 * The relayout only runs when the count actually changes, which is twice over a
 * drag across the threshold.
 */
class ClassGrid : public QWidget {
	struct Entry { QWidget *label, *toggle; };

public:
	explicit ClassGrid(QWidget *parent = nullptr) : QWidget(parent)
	{
		grid = new QGridLayout(this);
		grid->setContentsMargins(0, 10, 0, 12);
		grid->setHorizontalSpacing(24);
		grid->setVerticalSpacing(10);
	}

	void addEntry(QWidget *label, QWidget *toggle)
	{
		entries.append(Entry{ label, toggle });
		relayout(cols);
	}

protected:
	void resizeEvent(QResizeEvent *e) override
	{
		QWidget::resizeEvent(e);
		/*
		 * Roughly two of the longest class name plus its controls. The
		 * threshold went up with the per-class effect combo: at 500 the
		 * two columns overlapped and the right one was drawn clipped,
		 * which is the same defect the screenshot pass caught before.
		 */
		const int want = width() >= 720 ? 2 : 1;
		if (want != cols)
			relayout(want);
	}

private:
	void relayout(int n)
	{
		cols = n;
		/* Takes the layout items, not the widgets: everything stays
		 * parented to this and is re-added below. */
		while (QLayoutItem *it = grid->takeAt(0))
			delete it;
		for (int c = 0; c < 4; c++)
			grid->setColumnStretch(c, 0);

		int row = 0, cell = 0;
		for (const Entry &e : entries) {
			grid->addWidget(e.label, row, cell * 2);
			grid->addWidget(e.toggle, row, cell * 2 + 1,
			                Qt::AlignRight);
			if (++cell == cols) { cell = 0; row++; }
		}
		for (int c = 0; c < cols; c++)
			grid->setColumnStretch(c * 2, 1);
	}

	QGridLayout *grid;
	QList<Entry> entries;
	int cols = 2;
};

/*
 * A group of rows: a letterspaced heading, then a rounded panel holding the
 * settings themselves. Rows are separated by a hairline, which is what stops a
 * column of similar-looking rows reading as one block of text.
 */
struct Card {
	QVBoxLayout *box = nullptr;   /* where rows go */
	int          rows = 0;
};

Card *makeCard(QVBoxLayout *page, const QString &heading,
               const QString &note = QString())
{
	if (!heading.isEmpty()) {
		auto *h = new QLabel(heading.toUpper());
		h->setObjectName("sectionHeader");
		QFont f = h->font();
		f.setPointSizeF(f.pointSizeF() * 0.82);
		h->setFont(f);
		page->addWidget(h);
	}

	auto *frame = new QFrame;
	frame->setObjectName("card");
	auto *outer = new QVBoxLayout(frame);
	outer->setContentsMargins(16, 4, 16, 4);
	outer->setSpacing(0);

	if (!note.isEmpty()) {
		auto *n = new QLabel(note);
		n->setObjectName("note");
		n->setWordWrap(true);
		QFont f = n->font();
		f.setPointSizeF(f.pointSizeF() * 0.92);
		n->setFont(f);
		n->setContentsMargins(0, 10, 0, 2);
		outer->addWidget(n);
	}

	page->addWidget(frame);

	auto *c = new Card;
	c->box = outer;
	return c;
}

/*
 * One setting. Title and description on the left, the control on the right.
 *
 * The description stays *visible* rather than moving into a tooltip -- these
 * settings do not explain themselves, and a hover tooltip is invisible to
 * anyone who does not already suspect there is something to learn. What changed
 * is that it no longer shares a line with the control, so it can wrap to
 * whatever height it needs. It used to be clipped mid-sentence whenever the
 * window was anything short of maximised, which is the same bug as the panel
 * being unreadable: a layout that could not give a row the height it asked for.
 */
void addRow(Card *c, const QString &title, const QString &desc, QWidget *control,
            Qt::Alignment align = Qt::AlignTop | Qt::AlignRight)
{
	if (c->rows++) {
		auto *sep = new QFrame;
		sep->setObjectName("sep");
		sep->setFixedHeight(1);
		c->box->addWidget(sep);
	}

	auto *row = new QHBoxLayout;
	row->setContentsMargins(0, 12, 0, 12);
	row->setSpacing(16);

	auto *text = new QVBoxLayout;
	text->setSpacing(3);
	auto *t = new QLabel(title);
	t->setObjectName("rowTitle");
	t->setWordWrap(true);
	text->addWidget(t);

	if (!desc.isEmpty()) {
		auto *d = new QLabel(desc);
		d->setObjectName("rowDesc");
		d->setWordWrap(true);
		QFont f = d->font();
		f.setPointSizeF(f.pointSizeF() * 0.92);
		d->setFont(f);
		text->addWidget(d);
	}

	row->addLayout(text, 1);
	if (control)
		row->addWidget(control, 0, align);
	c->box->addLayout(row);
}

/* One line of the file, with the parsed key/value if it is a setting. */
struct Line {
	QString raw;
	QString key;      /* empty for comments, blanks and section headers */
	QString value;
	QString comment;  /* trailing "# ..." including the hash, if any */
	QString section;  /* the [section] this line falls under, lowercased */
};

QList<Line> g_lines;
QString     g_path;

/*
 * Identity of the copy on disk that g_lines was last read from -- or last
 * written to. Anything else on disk is somebody else's edit and has to be
 * picked up.
 *
 * This replaces a "we are writing, ignore the next notification" flag that was
 * cleared on a 300 ms timer. That flag could not tell our own write from one
 * the core made while it was still set, which is exactly what a model swap
 * does: the GUI writes `model`, and the core answers a moment later by
 * rewriting [classes] with the new taxonomy. Comparing what is on disk against
 * what we hold has no window to fall into.
 */
qint64 g_stamp_ms = -1;
qint64 g_stamp_sz = -1;

void stampFile()
{
	const QFileInfo fi(g_path);
	g_stamp_ms = fi.exists() ? fi.lastModified().toMSecsSinceEpoch() : -1;
	g_stamp_sz = fi.exists() ? fi.size() : -1;
}

/* Does disk hold something other than what we last read or wrote? */
bool diskDiffers()
{
	const QFileInfo fi(g_path);
	if (!fi.exists())
		return false;
	return fi.lastModified().toMSecsSinceEpoch() != g_stamp_ms ||
	       fi.size() != g_stamp_sz;
}

/*
 * Values in this file may carry a trailing comment. The core's parser strips
 * them (it did not always, and a boolean like `enabled = true # switch` used to
 * parse as false and silently disable censoring), so the GUI must strip them
 * the same way -- and put them back on save, or every edit would erase the
 * documentation the core wrote.
 */
void splitComment(const QString &in, QString &value, QString &comment)
{
	int hash = in.indexOf('#');
	int semi = in.indexOf(';');
	int cut  = (hash < 0) ? semi : (semi < 0 ? hash : qMin(hash, semi));
	if (cut < 0) {
		value = in.trimmed();
		comment.clear();
	} else {
		value   = in.left(cut).trimmed();
		comment = in.mid(cut);
	}
}

bool truthy(const QString &v)
{
	const QString t = v.trimmed().toLower();
	return t == "true" || t == "yes" || t == "on" || t == "1";
}

bool load(const QString &path)
{
	QFile f(path);
	if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
		return false;

	g_lines.clear();
	QString section;
	QTextStream in(&f);
	while (!in.atEnd()) {
		Line l;
		l.raw = in.readLine();

		const QString t = l.raw.trimmed();
		if (t.startsWith('[') && t.endsWith(']'))
			section = t.mid(1, t.length() - 2).trimmed().toLower();
		l.section = section;

		if (!t.isEmpty() && !t.startsWith('#') && !t.startsWith(';') &&
		    !t.startsWith('[')) {
			int eq = l.raw.indexOf('=');
			if (eq > 0) {
				l.key = l.raw.left(eq).trimmed();
				splitComment(l.raw.mid(eq + 1), l.value, l.comment);
			}
		}
		g_lines.append(l);
	}
	f.close();
	stampFile();
	return true;
}

void setValue(const QString &key, const QString &v)
{
	for (Line &l : g_lines)
		if (l.key == key) {
			l.value = v;
			return;
		}
}

QString value(const QString &key)
{
	for (const Line &l : g_lines)
		if (l.key == key)
			return l.value;
	return QString();
}

bool has(const QString &key) { return !value(key).isNull(); }

/*
 * Hand-drawn regions live in their own section and are addressed by position,
 * not by name -- the keys are labels the core writes as r1, r2, and nothing
 * reads them back. So these work on the section rather than through
 * setValue(), which looks a key up across the whole file and would happily
 * find an `r1` somewhere else.
 */
int regionCount()
{
	int n = 0;
	for (const Line &l : g_lines)
		if (l.section == QLatin1String("regions") && !l.key.isEmpty())
			n++;
	return n;
}

/*
 * Drop every drawn region, leaving the section and its documentation behind.
 *
 * Removing the header too would be tidier and wrong: the core writes it back
 * on the next rewrite anyway, and in the meantime the file would have no sign
 * that regions are a thing it can hold.
 */
void clearRegions()
{
	for (int i = g_lines.size() - 1; i >= 0; i--)
		if (g_lines[i].section == QLatin1String("regions") &&
		    !g_lines[i].key.isEmpty())
			g_lines.removeAt(i);
}

/*
 * Class names, tidied for reading.
 *
 * Two taxonomies need this for different reasons. The NudeNet-derived ones are
 * long but legible, so they only need reflowing: FEMALE_BREAST_EXPOSED becomes
 * "Female breast (exposed)". booru_yolo's are six-character identifiers -- SHLD,
 * NOPAN, VSPLT -- which are not guessable at all, so they get spelled out from
 * the author's own semantics (models/README.md). A name not in the table falls
 * through to the reflow, which is what keeps this working for a model we have
 * never seen.
 */
struct Pretty { const char *key, *label; };
const Pretty PRETTY[] = {
	{ "HEAD",   "Head" },          { "BUST",   "Breasts (clothed)" },
	{ "BOOB",   "Breasts (bare)" },{ "SIDEB",  "Side breast" },
	{ "SHLD",   "Shoulders" },     { "BELLY",  "Belly" },
	{ "NOPAN",  "Groin (bare)" },  { "BUTT",   "Buttocks (clothed)" },
	{ "ASS",    "Buttocks (bare)" },
	{ "HIP",    "Hips" },          { "WING",   "Wings" },
	{ "FERAL",  "Feral body" },
	{ "SPLIT",  "Legs apart" },    { "SPRD",   "Legs spread" },
	{ "VSPLT",  "Legs apart (front-on)" },
	{ "VSPRD",  "Legs spread (front-on)" },
	{ "JACKO",  "Presenting pose" },
	{ "JACKX",  "Presenting pose (legs crossed)" },
	{ "HDRAGO", "Dragon head" },   { "HPONY",  "Pony head" },
	{ "HFOX",   "Fox head" },      { "HRABB",  "Rabbit head" },
	{ "HCAT",   "Cat head" },      { "HBEAR",  "Bear head" },
	{ "HHORSE", "Horse head" },    { "HBIRD",  "Bird head" },
	{ "EYE",    "Eyes" },          { "FACE",   "Face" },
};

QString prettyName(const QString &key)
{
	for (const Pretty &p : PRETTY)
		if (key.compare(QLatin1String(p.key), Qt::CaseInsensitive) == 0)
			return QString::fromLatin1(p.label);

	/*
	 * FEMALE_BREAST_EXPOSED -> "Female breast (exposed)".
	 *
	 * Splitting on '_' is not enough, because one of the taxonomies writes
	 * the same names without separators -- FEMALEBREASTEXPOSED -- and that
	 * came out as "Femalebreast (exposed)". So tokenise against the words
	 * these names are actually built from, longest match first (or MALE
	 * would eat the tail of FEMALE). If the whole name resolves, use the
	 * words; if any of it does not, this is a taxonomy we do not know and
	 * splitting on '_' is the honest fallback.
	 */
	static const char *const WORDS[] = {
		"GENITALIA", "BUTTOCKS", "ARMPITS", "EXPOSED", "COVERED",
		"FEMALE", "BREAST", "BELLY", "MALE", "FACE", "FEET", "ANUS",
		"HEAD", "EYE",
	};

	QStringList parts;
	const QString flat = QString(key).remove('_').toUpper();
	for (int i = 0; i < flat.length(); ) {
		int best = -1, bestLen = 0;
		for (unsigned w = 0; w < sizeof WORDS / sizeof WORDS[0]; w++) {
			const QLatin1String word(WORDS[w]);
			const int len = word.size();
			if (len > bestLen && QStringView(flat).mid(i, len) == word) {
				best = int(w);
				bestLen = len;
			}
		}
		if (best < 0) {          /* unknown word: give up on the whole name */
			parts.clear();
			break;
		}
		parts << QString::fromLatin1(WORDS[best]);
		i += bestLen;
	}
	if (parts.isEmpty())
		parts = key.split('_', Qt::SkipEmptyParts);

	QString suffix;
	if (!parts.isEmpty()) {
		const QString last = parts.last().toUpper();
		if (last == "EXPOSED" || last == "COVERED") {
			suffix = " (" + last.toLower() + ")";
			parts.removeLast();
		}
	}

	QString s = parts.join(' ').toLower();
	if (!s.isEmpty())
		s[0] = s[0].toUpper();
	return s + suffix;
}

/*
 * The effects, with what each one actually does.
 *
 * This used to be a single paragraph describing all fifteen, attached to the
 * combo box -- around 900 characters that had to be read in full to learn about
 * whichever one was selected, and the first thing the old layout clipped. Only
 * the chosen effect's line is shown now, and the text is straight from the enum
 * in overlay.h so the two cannot drift apart.
 *
 * "Samples nothing" is the property worth stating: those effects do not read
 * the captured screen, so they keep covering even if the capture stalls, while
 * the rest fall back to a solid fill when there is no frame to sample.
 */
struct EffectInfo { const char *key, *desc; };
const EffectInfo EFFECTS[] = {
	{ "pixelate", "Block-average of the screen underneath." },
	{ "fill",     "A solid bar. Nothing shows through." },
	{ "blur",     "A wide box blur of the screen underneath." },
	{ "noise",    "Animated static. Samples nothing, so it keeps covering "
	              "even if the screen capture stalls." },
	{ "glitch",   "Torn strips of the pixelated screen." },
	{ "rgbshift", "Channel split over a pixelated base." },
	{ "tape",     "Hazard-striped bars spanning the whole screen through "
	              "each detected area. Every bar runs the full width, so this "
	              "looks best with only a few classes switched on." },
	{ "halftone", "Print-style dots over a pixelated base." },
	{ "sobel",    "Edge outlines of a pixelated base, on black." },
	{ "triangle", "Triangular mosaic of the screen underneath." },
	{ "cluster",  "A mass of discs. Samples nothing, so it keeps covering "
	              "even if the screen capture stalls." },
	{ "splatter", "Fused ink blobs. Samples nothing, so it keeps covering "
	              "even if the screen capture stalls." },
	{ "shape",    "Hexagonal tiling of the screen's own colours." },
	{ "caption",  "An opaque bar with the caption text on it. Samples nothing." },
	{ "wall",     "The area filled with repeated characters. Samples nothing." },
};

QString effectDesc(const QString &key)
{
	for (const EffectInfo &e : EFFECTS)
		if (key.compare(QLatin1String(e.key), Qt::CaseInsensitive) == 0)
			return QString::fromLatin1(e.desc);
	return QString();
}

/*
 * Which heading a class sits under. Grouping is by body area rather than by
 * model, so the panel reads the same whichever taxonomy is loaded -- the point
 * is to find the toggle you want without reading all of them.
 */
const char *const GROUPS[] = {
	"Faces and heads", "Chest", "Groin", "Buttocks and hips",
	"Poses", "Other",
};
const int N_GROUPS = int(sizeof GROUPS / sizeof GROUPS[0]);

int groupOf(const QString &key)
{
	const QString k = QString(key).remove('_').toUpper();

	static const char *const heads[] = {
		"HEAD", "HDRAGO", "HPONY", "HFOX", "HRABB", "HCAT", "HBEAR",
		"HHORSE", "HBIRD",
	};
	for (const char *h : heads)
		if (k == QLatin1String(h))       /* exact: HIP must not match here */
			return 0;
	if (k.contains("FACE") || k == "EYE")            return 0;
	if (k.contains("BREAST") || k == "BOOB" ||
	    k == "BUST" || k == "SIDEB")                 return 1;
	if (k.contains("GENITALIA") || k.contains("ANUS") ||
	    k == "NOPAN")                                return 2;
	if (k.contains("BUTTOCKS") || k == "BUTT" ||
	    k == "ASS" || k == "HIP")                    return 3;
	if (k == "SPLIT" || k == "SPRD" || k == "VSPLT" ||
	    k == "VSPRD" || k == "JACKO" || k == "JACKX" ||
	    k == "FERAL")                                return 4;
	return 5;
}

/*
 * Write via a temporary file and rename. The core may read this at any moment
 * -- it polls on mtime and does not lock -- so a partially written file would
 * be parsed as truncated, and a config that parses short is a config with
 * classes missing, i.e. censoring silently switched off. rename(2) is atomic,
 * so the core sees either the old file or the new one.
 */
bool save()
{
	const QString tmp = g_path + ".tmp";
	QFile f(tmp);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
		return false;

	QTextStream out(&f);
	for (const Line &l : g_lines) {
		if (l.key.isEmpty()) {
			out << l.raw << '\n';
		} else {
			/* Keep the original key column so the file stays as
			 * legible by hand as the core writes it. */
			int eq = l.raw.indexOf('=');
			QString lhs = (eq > 0) ? l.raw.left(eq) : (l.key + " ");
			out << lhs << "= " << l.value;
			if (!l.comment.isEmpty())
				out << "   " << l.comment;
			out << '\n';
		}
	}
	out.flush();
	f.close();

	/*
	 * ::rename, not QFile::rename, which refuses an existing destination and
	 * so had to be preceded by QFile::remove. That pair left a window with no
	 * config on disk at all -- the very thing the comment above says must not
	 * happen -- and it guaranteed the file watch was dropped on every save,
	 * because the inode being watched was deleted rather than replaced.
	 * rename(2) replaces in one step and the path never stops resolving.
	 */
	const bool ok = ::rename(QFile::encodeName(tmp).constData(),
	                         QFile::encodeName(g_path).constData()) == 0;
	if (!ok)
		QFile::remove(tmp);

	/* Ours, so the watcher and the poll below both know not to react to it. */
	stampFile();
	return ok;
}

} /* namespace */

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	/*
	 * Fusion, then our own palette on top. The native style is left behind
	 * deliberately: it draws frames and padding of its own that the rules in
	 * styleSheet() cannot predict, and a panel that is half themed by us and
	 * half by the desktop looks worse than either done alone.
	 */
	QApplication::setStyle(QStyleFactory::create("Fusion"));
	{
		QPalette p;
		p.setColor(QPalette::Window,          QColor(col::BG));
		p.setColor(QPalette::WindowText,      QColor(col::TEXT));
		p.setColor(QPalette::Base,            QColor(col::INPUT));
		p.setColor(QPalette::AlternateBase,   QColor(col::CARD));
		p.setColor(QPalette::Text,            QColor(col::TEXT));
		p.setColor(QPalette::Button,          QColor("#232329"));
		p.setColor(QPalette::ButtonText,      QColor(col::TEXT));
		p.setColor(QPalette::Highlight,       QColor(col::ACCENT));
		p.setColor(QPalette::HighlightedText, QColor("#ffffff"));
		p.setColor(QPalette::ToolTipBase,     QColor(col::INPUT));
		p.setColor(QPalette::ToolTipText,     QColor(col::TEXT));
		p.setColor(QPalette::Disabled, QPalette::Text,       QColor("#6a6a72"));
		p.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#6a6a72"));
		p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#6a6a72"));
		QApplication::setPalette(p);
	}
	/*
	 * Pick a proportional face rather than inheriting the desktop's default,
	 * which on this machine is a monospace one -- and a settings panel set in
	 * a terminal font reads as debug output whatever else is done to it. The
	 * list is preference order; whichever is installed first wins, and if
	 * none is, Qt's own default stands and nothing is worse than before.
	 */
	QString family = QApplication::font().family();
	{
		const char *want[] = { "Inter", "Noto Sans", "Open Sans", "Roboto",
		                       "Cantarell", "DejaVu Sans", "Liberation Sans" };
		const QStringList have = QFontDatabase::families();
		for (const char *fam : want) {
			if (!have.contains(QLatin1String(fam), Qt::CaseInsensitive))
				continue;
			family = QString::fromLatin1(fam);
			QFont f{ family };
			f.setPointSizeF(10.0);
			QApplication::setFont(f);
			break;
		}
	}
	app.setStyleSheet(styleSheet(family));

	g_path = (argc > 1) ? QString::fromLocal8Bit(argv[1])
	                    : QString(DEFAULT_CONFIG);

	QWidget win;
	win.setObjectName("root");
	win.setWindowTitle("BetaCensor");

	auto *root = new QVBoxLayout(&win);
	root->setContentsMargins(0, 0, 0, 0);
	root->setSpacing(0);

	auto *status = new QLabel;
	status->setObjectName("status");
	status->setWordWrap(true);

	if (!load(g_path)) {
		auto *pad = new QVBoxLayout;
		pad->setContentsMargins(24, 24, 24, 24);
		status->setText(QString("Cannot read %1.\n\nStart the censor once so it "
		                        "writes a config, or pass the path as an "
		                        "argument.").arg(QFileInfo(g_path).absoluteFilePath()));
		pad->addWidget(status);
		root->addLayout(pad);
		win.resize(460, 160);
		win.show();
		return app.exec();
	}

	/*
	 * Writes are debounced. A spin box emits on every step, and each write
	 * costs the core a reload; 200 ms is well under the 250 ms poll, so a
	 * change still lands on the very next tick.
	 */
	auto *debounce = new QTimer(&win);
	debounce->setSingleShot(true);
	debounce->setInterval(200);
	QObject::connect(debounce, &QTimer::timeout, [status] {
		status->setText(save() ? "Saved." : "Save FAILED -- check permissions.");
	});
	auto touch = [debounce] { debounce->start(); };

	/*
	 * Pulling the file back into the widgets. Every control registers how to
	 * re-read its own value, because the file changes underneath us for
	 * reasons that are not the user typing here: the core rewrites it on a
	 * model swap, and it is meant to stay hand-editable. Only the class
	 * checkboxes used to be refreshed, so an external edit to the effect or
	 * the sensitivity left the panel showing something that was no longer
	 * true -- and the next click here wrote the stale value straight back.
	 */
	QList<std::function<void()>> refreshers;

	/* --- chrome --------------------------------------------------------- */
	{
		auto *bar = new QHBoxLayout;
		bar->setContentsMargins(18, 14, 18, 10);
		auto *a = new QLabel("BETA");
		a->setObjectName("wordmarkAccent");
		auto *b = new QLabel("CENSOR");
		b->setObjectName("wordmark");
		for (QLabel *l : { a, b }) {
			QFont f = l->font();
			f.setPointSizeF(f.pointSizeF() * 1.15);
			l->setFont(f);
		}
		bar->addWidget(a);
		bar->addWidget(b);
		bar->addStretch();
		root->addLayout(bar);
	}

	auto *tabs = new QTabWidget;
	tabs->setDocumentMode(true);
	root->addWidget(tabs, 1);

	/*
	 * Each tab scrolls independently, which is the structural half of the
	 * fix. Everything used to live in one column with only the class list
	 * inside a scroll area, so the settings above it took whatever height
	 * they wanted and squeezed the classes into a strip too short to show a
	 * single checkbox -- while their own descriptions were clipped
	 * mid-sentence for want of the room they had just taken. Nothing here
	 * competes for height with anything on another tab.
	 */
	auto newTab = [tabs](const QString &name) {
		auto *page = new QWidget;
		auto *v = new QVBoxLayout(page);
		v->setContentsMargins(18, 16, 18, 18);
		v->setSpacing(16);

		auto *sc = new QScrollArea;
		sc->setWidget(page);
		sc->setWidgetResizable(true);
		sc->setFrameShape(QFrame::NoFrame);
		/* Wrapping labels compute their height from their width, so the
		 * width must never be decided by them: a horizontal scrollbar
		 * here would let a long description widen the page instead of
		 * wrapping, which is how the text came to be cut off. */
		sc->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		tabs->addTab(sc, name);
		return v;
	};

	/* --- tab: censoring -------------------------------------------------- */
	auto *tCensor = newTab("Censoring");

	if (has("enabled")) {
		auto *cb = new Toggle;
		cb->setChecked(truthy(value("enabled")));
		QObject::connect(cb, &QCheckBox::toggled, [touch](bool on) {
			setValue("enabled", on ? "true" : "false");
			touch();
		});
		Card *c = makeCard(tCensor, QString());
		addRow(c, "Censoring enabled",
		       "Master switch. When off, the screen is shown untouched and "
		       "nothing is covered.", cb);
		refreshers << [cb] {
			QSignalBlocker b(cb);
			cb->setChecked(truthy(value("enabled")));
		};
	}

	{
		Card *c = makeCard(tCensor, "Mode");

		if (has("reverse")) {
			auto *cb = new Toggle;
			cb->setChecked(truthy(value("reverse")));
			QObject::connect(cb, &QCheckBox::toggled, [touch](bool on) {
				setValue("reverse", on ? "true" : "false");
				touch();
			});
			addRow(c, "Reverse (show instead of hide)",
			       "Inverts the whole program: the screen is covered with the "
			       "chosen effect and the detected areas are the only parts "
			       "left visible. This deliberately reveals what would normally "
			       "be hidden, so it switches itself off whenever the panic key "
			       "is used or the pipeline loses the screen.\n\n"
			       "A class given an effect of its own under What to cover is "
			       "the exception: it stays covered with that effect instead of "
			       "being revealed. That is how you reveal a body but keep the "
			       "face censored. Classes left on Default are revealed.", cb);
			refreshers << [cb] {
				QSignalBlocker b(cb);
				cb->setChecked(truthy(value("reverse")));
			};
		}

		if (has("eye_bar")) {
			auto *cb = new Toggle;
			cb->setChecked(truthy(value("eye_bar")));
			QObject::connect(cb, &QCheckBox::toggled, [touch](bool on) {
				setValue("eye_bar", on ? "true" : "false");
				touch();
			});
			addRow(c, "Cover eyes only, not whole faces",
			       "When a face is detected, cover the eyes rather than the "
			       "whole face. With an eye-locating model set on this tab the "
			       "bar sits on the eyes themselves and turns with a tilted "
			       "head; without one it is a band across the middle of the "
			       "face, which covers the eyes at any angle but covers more. "
			       "Needs a face class enabled under What to cover.", cb);
			refreshers << [cb] {
				QSignalBlocker b(cb);
				cb->setChecked(truthy(value("eye_bar")));
			};

			/*
			 * What the bar itself is drawn with, and whether there
			 * is a separate bar at all. "None" is the behaviour the
			 * toggle above has always had -- the band replaces the
			 * face cover -- so leaving this alone changes nothing.
			 */
			auto *ecombo = new QComboBox;
			ecombo->addItem("None (the face class's own effect)",
			                QString("off"));
			for (const EffectInfo &e : EFFECTS)
				ecombo->addItem(QLatin1String(e.key),
				                QLatin1String(e.key));
			{
				int ei = ecombo->findData(value("eye_bar_effect"));
				ecombo->setCurrentIndex(ei >= 0 ? ei : 0);
			}
			QObject::connect(ecombo, &QComboBox::currentTextChanged,
			                 [ecombo, touch](const QString &) {
				setValue("eye_bar_effect",
				         ecombo->currentData().toString());
				touch();
			});
			addRow(c, "Effect for the eye bar",
			       "What the bar itself is drawn with, whatever is "
			       "happening to the rest of the face — so the face can be "
			       "wall-of-text while the eyes get a solid bar, or the "
			       "face can be left alone entirely and only the bar drawn. "
			       "On None the bar takes the effect of the face class it "
			       "came from.", ecombo);
			refreshers << [ecombo] {
				QSignalBlocker b(ecombo);
				int i = ecombo->findData(value("eye_bar_effect"));
				ecombo->setCurrentIndex(i >= 0 ? i : 0);
			};

			/*
			 * The other half, which used to be welded to the combo
			 * above: naming an effect for the bar was also the only
			 * way to say "cover the face", so a bar with an effect
			 * of its own over an otherwise untouched face -- an eye
			 * bar in the ordinary sense -- could not be asked for.
			 */
			if (has("eye_bar_face")) {
				auto *fb = new Toggle;
				fb->setChecked(truthy(value("eye_bar_face")));
				QObject::connect(fb, &QCheckBox::toggled,
				                 [touch](bool on) {
					setValue("eye_bar_face",
					         on ? "true" : "false");
					touch();
				});
				addRow(c, "Cover the face under the bar",
				       "On, the face is covered with its own effect "
				       "and the bar is drawn on top. Off, only the "
				       "bar is drawn and the rest of the face is "
				       "left as it is — which is what you want for a "
				       "plain eye bar. Off is the safer-looking "
				       "setting but covers less, so it is the one "
				       "place here where turning something off "
				       "reveals a face that was covered before.", fb);
				refreshers << [fb] {
					QSignalBlocker b(fb);
					fb->setChecked(truthy(value("eye_bar_face")));
				};
			}
		}

		if (has("join")) {
			auto *cb = new Toggle;
			cb->setChecked(truthy(value("join")));
			QObject::connect(cb, &QCheckBox::toggled, [touch](bool on) {
				setValue("join", on ? "true" : "false");
				touch();
			});
			addRow(c, "Join overlapping covers",
			       "Detections of parts of one body overlap constantly — a "
			       "torso, a chest and a belly are three boxes over one person "
			       "— and drawing all three stacks edges and seams inside "
			       "what is really one region. This merges them into a single "
			       "cover, which always includes everything the boxes it "
			       "replaced covered, so it can never uncover anything.", cb);
			refreshers << [cb] {
				QSignalBlocker b(cb);
				cb->setChecked(truthy(value("join")));
			};
		}
	}

	if (has("model")) {
		/*
		 * Model picker. Swapping the detector is the one setting here
		 * that changes what the *other* settings mean: each model has
		 * its own taxonomy, so the class list is rebuilt when this
		 * changes. Offer only models that actually exist rather than a
		 * free-text path -- a typo would otherwise be discovered by the
		 * core at swap time, not here.
		 */
		const QString cur = value("model");
		auto *combo = new QComboBox;
		combo->setMinimumWidth(240);

		QStringList dirs;
		if (!cur.isEmpty())
			dirs << QFileInfo(cur).path();
		dirs << "vendor";
		dirs.removeDuplicates();

		QStringList found;
		for (const QString &d : dirs) {
			QDir dir(d);
			for (const QString &f :
			     dir.entryList({ "*.onnx" }, QDir::Files, QDir::Name)) {
				const QString rel = dir.filePath(f);
				if (!found.contains(rel))
					found << rel;
			}
		}
		/* Whatever is loaded stays selectable even if the scan missed
		 * it -- the running core is the authority on what exists. */
		if (!cur.isEmpty() && !found.contains(cur))
			found.prepend(cur);

		for (const QString &m : found)
			combo->addItem(QFileInfo(m).fileName(), m);
		int idx = combo->findData(cur);
		if (idx >= 0)
			combo->setCurrentIndex(idx);

		QObject::connect(combo, &QComboBox::currentTextChanged,
		                 [combo, touch](const QString &) {
			setValue("model", combo->currentData().toString());
			touch();
		});

		Card *c = makeCard(tCensor, "Detector");
		addRow(c, "Model",
		       "Which model does the detecting. Swapping takes about a second, "
		       "during which the screen is blanked rather than left uncovered. "
		       "Each model has its own class list, so What to cover is rebuilt "
		       "with it and each model remembers its own settings.", combo);
		refreshers << [combo] {
			QSignalBlocker b(combo);
			int i = combo->findData(value("model"));
			if (i >= 0)
				combo->setCurrentIndex(i);
		};

		/*
		 * The eye-locating model, on the same card because it is the
		 * other half of "what is loaded". It never censors anything on
		 * its own: it only says where the eyes are inside faces the
		 * detector above already found, so the class list does not
		 * change with it and there is nothing to rebuild.
		 *
		 * "None" is a real choice rather than an empty field, because
		 * without one the eye bar silently becomes a band across the
		 * whole face -- which works, and looks like a different feature.
		 */
		auto *lcombo = new QComboBox;
		lcombo->setMinimumWidth(240);
		lcombo->addItem("None (band across the face)", QString());
		for (const QString &m : found)
			lcombo->addItem(QFileInfo(m).fileName(), m);
		const QString lcur = value("landmark_model");
		if (!lcur.isEmpty() && lcombo->findData(lcur) < 0)
			lcombo->addItem(QFileInfo(lcur).fileName(), lcur);
		{
			int li = lcombo->findData(lcur);
			lcombo->setCurrentIndex(li >= 0 ? li : 0);
		}
		QObject::connect(lcombo, &QComboBox::currentTextChanged,
		                 [lcombo, touch](const QString &) {
			setValue("landmark_model",
			         lcombo->currentData().toString());
			touch();
		});
		addRow(c, "Eye-locating model",
		       "Optional second model, used only to find the eyes inside a "
		       "detected face so the eye bar can sit on them and turn with a "
		       "tilted head. It never covers anything by itself. Without one, "
		       "\"Cover eyes only\" falls back to a band across the middle of "
		       "the face, which covers the eyes at any angle but covers more. "
		       "Needs a model trained on the kind of faces you are covering.",
		       lcombo);
		refreshers << [lcombo] {
			QSignalBlocker b(lcombo);
			int i = lcombo->findData(value("landmark_model"));
			lcombo->setCurrentIndex(i >= 0 ? i : 0);
		};

		/*
		 * The outline model. Third on the same card for the same reason
		 * as the second: it is part of "what is loaded" and it censors
		 * nothing on its own. Unlike the other two it can only ever make
		 * covers *smaller*, which the description has to say plainly --
		 * it is the one control here that trades coverage for looks.
		 */
		if (has("seg_model")) {
			auto *scombo = new QComboBox;
			scombo->setMinimumWidth(240);
			scombo->addItem("None (covers stay rectangular)", QString());
			for (const QString &m : found)
				scombo->addItem(QFileInfo(m).fileName(), m);
			const QString scur = value("seg_model");
			if (!scur.isEmpty() && scombo->findData(scur) < 0)
				scombo->addItem(QFileInfo(scur).fileName(), scur);
			{
				int si = scombo->findData(scur);
				scombo->setCurrentIndex(si >= 0 ? si : 0);
			}
			QObject::connect(scombo, &QComboBox::currentTextChanged,
			                 [scombo, touch](const QString &) {
				setValue("seg_model",
				         scombo->currentData().toString());
				touch();
			});
			addRow(c, "Outline model",
			       "Optional model that traces the subject's outline, so covers "
			       "follow their shape instead of being rectangles. Needs a "
			       "segmentation model (one with \"-seg\" in the name).\n\n"
			       "This is the only setting here that covers *less*: anything "
			       "the outline misses is left showing. The margin below exists "
			       "to absorb that, and without a model loaded every cover is "
			       "the full rectangle it has always been.",
			       scombo);
			refreshers << [scombo] {
				QSignalBlocker b(scombo);
				int i = scombo->findData(value("seg_model"));
				scombo->setCurrentIndex(i >= 0 ? i : 0);
			};
		}

		if (has("seg_margin")) {
			auto *sb = new QDoubleSpinBox;
			sb->setRange(0.0, 25.0);
			sb->setSingleStep(0.5);
			sb->setDecimals(2);
			sb->setSuffix(" %");
			sb->setValue(value("seg_margin").toDouble());
			QObject::connect(sb, &QDoubleSpinBox::valueChanged,
			                 [touch](double v) {
				setValue("seg_margin", QString::number(v, 'f', 2));
				touch();
			});
			addRow(c, "Outline margin",
			       "How far the traced outline is grown before it clips "
			       "anything, as a percentage of the screen's short side. The "
			       "trace is approximate in both directions, so without a "
			       "margin a rim of the subject ends up outside it. Raise this "
			       "if you see edges showing; lower it for a tighter shape.",
			       sb);
			refreshers << [sb] {
				QSignalBlocker b(sb);
				sb->setValue(value("seg_margin").toDouble());
			};
		}
	}
	/*
	 * Hand-drawn regions.
	 *
	 * The backstop for what the detector does not find. Everything else on
	 * this tab tunes a model's judgement; this is the row that does not ask
	 * a model anything, which is what makes it worth having -- a subject
	 * scoring under the confidence floor is uncovered, and nothing in the
	 * pipeline is able to notice.
	 */
	if (has("draw")) {
		Card *c = makeCard(tCensor, "Areas you draw yourself");

		auto *drawBtn  = new QPushButton;
		auto *clearBtn = new QPushButton;
		drawBtn->setMinimumWidth(150);
		clearBtn->setMinimumWidth(150);

		/*
		 * The button reports the core's answer, not the click. Drawing
		 * mode is a request -- the core turns the flag off again when
		 * the drawing ends, times out, or a panic cancels it -- so a
		 * button that latched on its own press would sit there saying
		 * "Stop drawing" over a mode that had already stopped.
		 */
		auto paintDraw = [drawBtn, clearBtn] {
			const bool on = truthy(value("draw"));
			drawBtn->setText(on ? "Stop drawing" : "Draw an area");
			const int n = regionCount();
			clearBtn->setEnabled(n > 0);
			clearBtn->setText(n > 0
				? QString("Clear %1 area%2").arg(n).arg(n == 1 ? "" : "s")
				: QString("Nothing to clear"));
		};

		QObject::connect(drawBtn, &QPushButton::clicked, [paintDraw] {
			const bool on = truthy(value("draw"));
			setValue("draw", on ? "false" : "true");
			/*
			 * Saved immediately rather than debounced. The other
			 * controls here are values being tuned; this one hands
			 * the pointer over, and 200 ms of nothing happening
			 * after pressing it reads as the button not working.
			 */
			save();
			paintDraw();
		});

		QObject::connect(clearBtn, &QPushButton::clicked,
		                 [paintDraw, &win] {
			const int n = regionCount();
			if (n <= 0)
				return;
			/*
			 * Asked about, unlike every other control here. This is
			 * the one button that removes covered area, it does it
			 * to work that was placed by hand rather than typed,
			 * and there is nothing to undo it with.
			 */
			if (QMessageBox::question(
				&win, "Clear drawn areas",
				QString("Remove %1 hand-drawn area%2? "
				        "Whatever they were covering will be left "
				        "to the detector.")
					.arg(n).arg(n == 1 ? "" : "s"),
				QMessageBox::Yes | QMessageBox::No,
				QMessageBox::No) != QMessageBox::Yes)
				return;
			clearRegions();
			save();
			paintDraw();
		});

		paintDraw();

		{
			auto *row = new QWidget;
			auto *h = new QHBoxLayout(row);
			h->setContentsMargins(0, 0, 0, 0);
			h->setSpacing(8);
			h->addWidget(drawBtn);
			h->addWidget(clearBtn);
			addRow(c, "Draw or trace an area to cover",
			       "Covers an area no matter what the detector sees, for "
			       "the things it misses. Press the button, then either "
			       "hold the mouse down and trace round what you want "
			       "covered, or click once at each of two opposite corners "
			       "for a rectangle (Enter does the same as a click). A "
			       "traced shape is only taken if you finish where you "
			       "started -- it fills in as soon as it would be "
			       "accepted, so let go while it is solid. Escape abandons "
			       "what you are drawing, and Escape again finishes. While "
			       "drawing, this screen takes the mouse and keyboard; it "
			       "gives them back when you finish, or when you press "
			       "this button again — which works from here whatever "
			       "the drawing screen is doing, so take as long as you "
			       "like over a trace.", row);
		}

		refreshers << paintDraw;
	}

	tCensor->addStretch(1);

	/* --- tab: effect ----------------------------------------------------- */
	auto *tEffect = newTab("Effect");

	if (has("effect")) {
		Card *c = makeCard(tEffect, "How covers are drawn");

		auto *combo = new QComboBox;
		combo->setMinimumWidth(170);
		for (const EffectInfo &e : EFFECTS)
			combo->addItem(QString::fromLatin1(e.key));
		{
			const QString cur = value("effect").toLower();
			combo->setCurrentText(combo->findText(cur) >= 0 ? cur
			                                                : "pixelate");
		}
		addRow(c, "Effect",
		       "How a detected area is hidden.", combo);

		/* The chosen effect's own description, on its own row so it has
		 * the full width to wrap into. */
		auto *what = new QLabel;
		what->setObjectName("rowDesc");
		what->setWordWrap(true);
		what->setContentsMargins(0, 0, 0, 12);
		{
			QFont f = what->font();
			f.setPointSizeF(f.pointSizeF() * 0.92);
			what->setFont(f);
		}
		c->box->addWidget(what);

		/*
		 * Heap-allocated, not a local captured by reference: showDesc
		 * outlives this scope -- it is held by the combo's connection
		 * and by the refresher list -- so a reference to a stack
		 * variable here would dangle the moment the tab was built.
		 */
		auto *strengthReload = new std::function<void()>;
		auto showDesc = [what, combo, strengthReload] {
			what->setText(effectDesc(combo->currentText()));
			if (*strengthReload)
				(*strengthReload)();
		};
		showDesc();
		QObject::connect(combo, &QComboBox::currentTextChanged,
		                 [touch, showDesc](const QString &t) {
			setValue("effect", t);
			showDesc();
			touch();
		});
		refreshers << [combo, showDesc] {
			QSignalBlocker b(combo);
			const QString v = value("effect").toLower();
			if (combo->findText(v) >= 0)
				combo->setCurrentText(v);
			showDesc();
		};

		/*
		 * One box, but it edits whichever effect's value is in play --
		 * the effects each keep their own, because the number means a
		 * different quantity to each of them. Its label says which,
		 * since a spin box that silently retargets is worse than no
		 * spin box.
		 */
		auto *sb = new QDoubleSpinBox;
		sb->setRange(1, 100);
		sb->setSingleStep(1);
		sb->setDecimals(1);
		sb->setFixedWidth(110);
		auto *coarse = new QLabel;
		coarse->setObjectName("rowTitle");
		coarse->setWordWrap(true);

		auto sKey = [combo] {
			return QString("strength_") + combo->currentText();
		};
		auto loadStrength = [sb, sKey, coarse, combo] {
			QSignalBlocker b(sb);
			sb->setValue(value(sKey()).toDouble());
			const QString e = combo->currentText();
			const bool used = (e == "pixelate" || e == "blur" ||
			                   e == "glitch"   || e == "rgbshift" ||
			                   e == "halftone" || e == "sobel" ||
			                   e == "triangle" || e == "shape");
			coarse->setText(used
				? QString("Coarseness of %1").arg(e)
				: QString("Coarseness (%1 does not use it)").arg(e));
			sb->setEnabled(used);
		};
		QObject::connect(sb, &QDoubleSpinBox::valueChanged,
		                 [touch, sKey](double v) {
			setValue(sKey(), QString::number(v, 'f', 1));
			touch();
		});

		{
			/* addRow() takes a title string; this one changes, so the
			 * label is built by hand and the row assembled around it. */
			auto *row = new QHBoxLayout;
			row->setContentsMargins(0, 12, 0, 12);
			row->setSpacing(16);
			auto *txt = new QVBoxLayout;
			txt->setSpacing(3);
			txt->addWidget(coarse);
			auto *d = new QLabel(
				"How coarse the effect is, as a percentage of the covered "
				"area's shorter side \u2014 block size, blur radius or "
				"facet size depending on the effect. Higher hides more. "
				"Each effect keeps its own value.");
			d->setObjectName("rowDesc");
			d->setWordWrap(true);
			QFont f = d->font();
			f.setPointSizeF(f.pointSizeF() * 0.92);
			d->setFont(f);
			txt->addWidget(d);
			auto *sep = new QFrame;
			sep->setObjectName("sep");
			sep->setFixedHeight(1);
			c->box->addWidget(sep);
			row->addLayout(txt, 1);
			row->addWidget(sb, 0, Qt::AlignTop | Qt::AlignRight);
			c->box->addLayout(row);
			c->rows++;
		}
		loadStrength();
		refreshers << loadStrength;
		*strengthReload = loadStrength;
	}

	if (has("censor_color") || has("outline") || has("shadow")) {
		Card *c = makeCard(tEffect, "The censor shape",
		                   "Decoration on the cover itself. The outline is drawn "
		                   "outward from each area and the shadow beyond it, so "
		                   "neither takes anything away from what is covered.");

		if (has("censor_color")) {
			/*
			 * Same three-way shape as the glyph colour: a kind, and
			 * a swatch that only means anything for one of them.
			 */
			auto *host = new QWidget;
			auto *row  = new QHBoxLayout(host);
			row->setContentsMargins(0, 0, 0, 0);
			row->setSpacing(8);
			auto *kind   = new QComboBox;
			auto *swatch = new QPushButton;
			kind->addItem("Black", "black");
			kind->addItem("Sampled from around it", "auto");
			kind->addItem("Pick a colour\u2026", "solid");
			swatch->setFixedWidth(78);
			row->addWidget(kind);
			row->addWidget(swatch);

			auto *held = new QColor(QColor("#202020"));
			auto paint = [swatch, held](bool on) {
				swatch->setVisible(on);
				swatch->setText(on ? held->name().toUpper() : QString());
				swatch->setStyleSheet(
					on ? QString("background:%1; color:%2;"
					             "border:1px solid #33333a;"
					             "border-radius:6px; font-weight:600;")
					     .arg(held->name(),
					          held->lightness() > 127 ? "#000" : "#fff")
					   : QString());
			};
			auto apply = [kind, held, touch, paint] {
				const QString k = kind->currentData().toString();
				paint(k == "solid");
				setValue("censor_color",
				         k == "solid" ? held->name() : k);
				touch();
			};
			QObject::connect(kind, &QComboBox::currentTextChanged,
			                 [apply] { apply(); });
			QObject::connect(swatch, &QPushButton::clicked,
			                 [held, apply, &win] {
				const QColor cc = QColorDialog::getColor(
					*held, &win, "Censor colour");
				if (cc.isValid()) { *held = cc; apply(); }
			});
			addRow(c, "Bar colour",
			       "Only affects the solid Fill effect \u2014 the others draw "
			       "their own colours. Sampled takes it from the pixels around "
			       "each area, so the bar sits into the picture instead of "
			       "reading as a flat black rectangle.", host);
			refreshers << [kind, held, paint] {
				QSignalBlocker b(kind);
				const QString v = value("censor_color").trimmed();
				QString k = "black";
				if (!v.compare("auto", Qt::CaseInsensitive)) {
					k = "auto";
				} else if (QColor::isValidColorName(v) &&
				           v.compare("black", Qt::CaseInsensitive)) {
					k = "solid";
					*held = QColor(v);
				}
				kind->setCurrentIndex(kind->findData(k));
				paint(k == "solid");
			};
		}

		auto addF = [&](const char *key, const QString &title,
		                const QString &desc, double lo, double hi,
		                double step, int dec) {
			if (!has(key))
				return;
			auto *sb = new QDoubleSpinBox;
			sb->setRange(lo, hi);
			sb->setSingleStep(step);
			sb->setDecimals(dec);
			sb->setValue(value(key).toDouble());
			sb->setFixedWidth(110);
			const QString k = key;
			QObject::connect(sb, &QDoubleSpinBox::valueChanged,
			                 [touch, k, dec](double v) {
				setValue(k, QString::number(v, 'f', dec));
				touch();
			});
			addRow(c, title, desc, sb);
			refreshers << [sb, k] {
				QSignalBlocker b(sb);
				sb->setValue(value(k).toDouble());
			};
		};

		addF("outline", "Outline width",
		     "Pixels of outline around every covered area, drawn outward. "
		     "0 switches it off.", 0, 40, 1, 1);

		if (has("outline_color")) {
			auto *btn = new QPushButton;
			btn->setFixedWidth(96);
			auto *held = new QColor(QColor(value("outline_color").trimmed()));
			if (!held->isValid())
				*held = QColor("#000000");
			auto paint = [btn, held] {
				btn->setText(held->name().toUpper());
				btn->setStyleSheet(
					QString("background:%1; color:%2;"
					        "border:1px solid #33333a;"
					        "border-radius:6px; font-weight:600;")
					.arg(held->name(),
					     held->lightness() > 127 ? "#000" : "#fff"));
			};
			QObject::connect(btn, &QPushButton::clicked,
			                 [held, paint, touch, &win] {
				const QColor cc = QColorDialog::getColor(
					*held, &win, "Outline colour");
				if (!cc.isValid())
					return;
				*held = cc;
				paint();
				setValue("outline_color", cc.name());
				touch();
			});
			paint();
			addRow(c, "Outline colour", QString(), btn);
			refreshers << [held, paint] {
				const QColor cc(value("outline_color").trimmed());
				if (cc.isValid())
					*held = cc;
				paint();
			};
		}

		addF("shadow", "Shadow size",
		     "Pixels of soft shadow beyond the outline. 0 switches it off.",
		     0, 60, 1, 1);
		addF("shadow_opacity", "Shadow strength",
		     "How dark the shadow is at its strongest.", 0, 1, 0.05, 2);
	}

	if (has("growth") || has("takeover_ms")) {
		Card *c = makeCard(tEffect, "Escalation",
		                   "Both only ever cover more, never less, so neither "
		                   "can expose anything that was covered before.");

		auto addE = [&](const char *key, const QString &title,
		                const QString &desc, double lo, double hi,
		                double step, int dec) {
			if (!has(key))
				return;
			auto *sb = new QDoubleSpinBox;
			sb->setRange(lo, hi);
			sb->setSingleStep(step);
			sb->setDecimals(dec);
			sb->setValue(value(key).toDouble());
			sb->setFixedWidth(110);
			const QString k = key;
			QObject::connect(sb, &QDoubleSpinBox::valueChanged,
			                 [touch, k, dec](double v) {
				setValue(k, QString::number(v, 'f', dec));
				touch();
			});
			addRow(c, title, desc, sb);
			refreshers << [sb, k] {
				QSignalBlocker b(sb);
				sb->setValue(value(k).toDouble());
			};
		};

		addE("growth", "Growth per second",
		     "Each cover creeps outward by this percentage for every second "
		     "the thing under it stays on screen. 0 switches it off.",
		     0, 200, 5, 1);
		addE("growth_max", "Growth limit",
		     "How far a cover may grow in total, as a percentage of its "
		     "original size.", 0, 400, 10, 1);
		addE("takeover_ms", "Take over the screen after",
		     "Milliseconds a detection must persist before the whole screen "
		     "is covered instead of just the detected areas. 0 switches it "
		     "off. Ignored in reverse mode, where covering everything would "
		     "reveal everything.", 0, 60000, 500, 0);
	}

	if (has("caption") || has("text_color")) {
		Card *c = makeCard(tEffect, "Lettering",
		                   "Used by the caption and wall-of-text effects. Only "
		                   "the lettering changes — the bar underneath is "
		                   "what actually covers, so nothing here can leave "
		                   "anything exposed.");

		if (has("caption")) {
			/*
			 * Free text, unlike the model path: there is nothing to
			 * typo into a failure. An unusable value degrades to the
			 * class name or to blank, and the bar covers either way.
			 *
			 * Applied as typed rather than on return. The core
			 * re-reads at 4 Hz, so the text on screen follows the
			 * keystrokes, which is the point of putting it here.
			 */
			auto *edit = new QLineEdit(value("caption"));
			edit->setMaxLength(60);
			edit->setPlaceholderText("(class name)");
			edit->setMinimumWidth(200);
			QObject::connect(edit, &QLineEdit::textChanged,
			                 [touch](const QString &t) {
				setValue("caption", t);
				touch();
			});
			addRow(c, "Caption text",
			       "Leave it empty and each bar is labelled with the class that "
			       "triggered it instead, which doubles as a live read-out of "
			       "what the detector thinks it is looking at. The wall repeats "
			       "the text to fill the area; with no text it draws random "
			       "letters.", edit);
			refreshers << [edit] {
				QSignalBlocker b(edit);
				const QString v = value("caption");
				if (edit->text() != v)
					edit->setText(v);
			};
		}

		if (has("text_color")) {
			/*
			 * Three kinds of value share one key: "auto", "rainbow",
			 * and a colour. A combo cannot hold the third and a plain
			 * line edit makes the first two undiscoverable, so it is
			 * both -- the combo chooses the kind, and the swatch
			 * beside it is only enabled for the kind that has a
			 * colour to pick.
			 */
			auto *host = new QWidget;
			auto *row  = new QHBoxLayout(host);
			row->setContentsMargins(0, 0, 0, 0);
			row->setSpacing(8);
			auto *kind   = new QComboBox;
			auto *swatch = new QPushButton;
			kind->addItem("Effect's own colour", "auto");
			kind->addItem("Rainbow wave (glows)", "rainbow");
			kind->addItem("Pick a colour…",  "solid");
			swatch->setFixedWidth(78);
			row->addWidget(kind);
			row->addWidget(swatch);

			/* The colour last picked, kept so flipping to rainbow and
			 * back does not lose it. */
			auto *held = new QColor(QColor("#40ffd0"));

			auto paintSwatch = [swatch, held](bool on) {
				/* Hidden rather than disabled: an empty grey box
				 * beside the combo reads as something broken,
				 * where nothing at all reads as "this kind of
				 * colour has no swatch to pick". */
				swatch->setVisible(on);
				swatch->setEnabled(on);
				swatch->setText(on ? held->name().toUpper() : QString());
				swatch->setStyleSheet(
					on ? QString("background:%1; color:%2;"
					             "border:1px solid #33333a;"
					             "border-radius:6px; font-weight:600;")
					     .arg(held->name(),
					          held->lightness() > 127 ? "#000" : "#fff")
					   : QString());
			};

			auto apply = [kind, held, touch, paintSwatch] {
				const QString k = kind->currentData().toString();
				const bool solid = (k == "solid");
				paintSwatch(solid);
				setValue("text_color", solid ? held->name() : k);
				touch();
			};

			QObject::connect(kind, &QComboBox::currentTextChanged,
			                 [apply] { apply(); });
			QObject::connect(swatch, &QPushButton::clicked,
			                 [held, apply, &win] {
				const QColor cc = QColorDialog::getColor(
					*held, &win, "Text colour");
				if (cc.isValid()) {
					*held = cc;
					apply();
				}
			});

			addRow(c, "Text colour",
			       "Rainbow wave sweeps the hue across the box and over time, "
			       "and adds a glow around the letters.", host);

			refreshers << [kind, held, paintSwatch] {
				QSignalBlocker b(kind);
				const QString v = value("text_color").trimmed();
				QString k = "auto";
				if (!v.compare("rainbow", Qt::CaseInsensitive) ||
				    !v.compare("rgb", Qt::CaseInsensitive) ||
				    !v.compare("wave", Qt::CaseInsensitive)) {
					k = "rainbow";
				} else if (QColor::isValidColorName(v)) {
					k = "solid";
					*held = QColor(v);
				}
				kind->setCurrentIndex(kind->findData(k));
				paintSwatch(k == "solid");
			};
		}
	}
	tEffect->addStretch(1);

	/* --- tab: detection -------------------------------------------------- */
	auto *tDetect = newTab("Detection");

	struct NumSpec {
		const char *key, *label, *tip;
		double lo, hi, step;
		int decimals;
	};
	static const NumSpec nums[] = {
		{ "eye_bar_pad", "Eye bar size",
		  "How far the eye bar extends past the eyes themselves. Only "
		  "used when an eye-locating model is set on the Censoring tab "
		  "-- without one the bar is a band across the whole face and "
		  "this does nothing. Raise it if the bar looks too small to "
		  "cover the eyes properly.", 0.5, 6.0, 0.05, 2 },
		{ "strength",   "Pixelation coarseness",
		  "Size of each pixelated block, as a percentage of the covered "
		  "area's shorter side. Higher is blockier and harder to see "
		  "through. Only affects the Pixelate effect.", 1, 100, 1, 1 },
		{ "confidence", "Detection sensitivity",
		  "How certain the detector must be before it covers something. "
		  "Lower covers more and occasionally covers something harmless; "
		  "higher covers only what it is sure about and will miss more. "
		  "This program is deliberately biased toward covering too much.",
		  0.05, 0.95, 0.05, 2 },
		{ "dilate",     "Extra padding around each area",
		  "Grows every covered area by this percentage, so content that "
		  "moves slightly between being detected and being drawn stays "
		  "covered. Raise it if edges peek out during fast movement.",
		  0, 100, 1, 1 },
		{ "sticky_ms",  "How long a cover lingers (milliseconds)",
		  "After the detector stops seeing something, its cover stays for "
		  "this long before fading. Too low and covers flicker on and off "
		  "when the detector briefly loses track; too high and covers hang "
		  "around over content that has already gone. 1000 = one second.",
		  0, 5000, 10, 0 },
		{ "delay_ms",   "Screen delay (milliseconds)",
		  "Shows the screen this far behind real time, so every frame is "
		  "censored using detections from both before and after it instead "
		  "of a guess at where things have moved. Covers then sit exactly "
		  "on what they hide, including through fast motion and for eye "
		  "bars. The price is that the picture lags by this much (audio "
		  "does not, and the mouse pointer probably does not either, so it "
		  "will be ahead of what it points at), which suits watching video "
		  "rather than playing games. 0 = off (live, no delay). Needs to "
		  "be longer than one detection pass; if it is not, the picture "
		  "holds still until the detector catches up, which the run "
		  "summary counts.", 0, 1000, 10, 0 },
		{ "smoothing",  "Steadiness of cover edges",
		  "How quickly a cover shrinks back when the area it is covering "
		  "gets smaller. Lower is steadier and less jittery, but covers "
		  "take longer to tighten up again; higher follows the detector "
		  "closely and looks jumpier. Growing is always instant, so a low "
		  "value never uncovers anything early.", 0.01, 1.0, 0.05, 2 },
	};

	{
		Card *c = makeCard(tDetect, "Tuning");
		for (const NumSpec &n : nums) {
			/* strength lives on the Effect tab, next to the effect it
			 * is the coarseness of. */
			if (!has(n.key) || !qstrcmp(n.key, "strength"))
				continue;
			auto *sb = new QDoubleSpinBox;
			sb->setRange(n.lo, n.hi);
			sb->setSingleStep(n.step);
			sb->setDecimals(n.decimals);
			sb->setValue(value(n.key).toDouble());
			sb->setFixedWidth(110);
			const QString key = n.key;
			const int dec = n.decimals;
			QObject::connect(sb, &QDoubleSpinBox::valueChanged,
			                 [touch, key, dec](double v) {
				setValue(key, QString::number(v, 'f', dec));
				touch();
			});
			addRow(c, n.label, n.tip, sb);
			refreshers << [sb, key] {
				QSignalBlocker b(sb);
				sb->setValue(value(key).toDouble());
			};
		}
	}

	/*
	 * The section grid. These reload live like everything else but had no
	 * controls at all until now, so the only way to change them was to edit
	 * the file by hand -- which is the same "hassle to use" as the layout.
	 */
	if (has("section_cols") || has("section_rows")) {
		Card *c = makeCard(tDetect, "Screen sections",
		                   "Split the screen into a grid and run the detector on "
		                   "each tile. Each tile is scaled down less than the "
		                   "whole screen would be, so small things stay big "
		                   "enough to recognise — at the cost of one "
		                   "inference per tile, which is the scarcest thing in "
		                   "the loop.");

		auto *cost = new QLabel;
		cost->setObjectName("note");
		cost->setWordWrap(true);
		{
			QFont f = cost->font();
			f.setPointSizeF(f.pointSizeF() * 0.92);
			cost->setFont(f);
		}

		auto showCost = [cost] {
			if (truthy(value("section_split"))) {
				const bool full = truthy(value("section_full_frame"));
				cost->setText(QString("Split mode: usually 2 inferences per frame on a "
				                      "widescreen%1.")
				              .arg(full ? ", plus one for the whole screen" : ""));
				return;
			}
			const int cols = qMax(1, value("section_cols").toInt());
			const int rows = qMax(1, value("section_rows").toInt());
			const bool full = truthy(value("section_full_frame")) &&
			                  (cols > 1 || rows > 1);
			const int n = cols * rows + (full ? 1 : 0);
			cost->setText(QString("Currently %1 inference%2 per frame.")
			              .arg(n).arg(n == 1 ? "" : "s"));
		};

		auto addInt = [&](const char *key, const QString &title,
		                  const QString &desc, int lo, int hi) {
			if (!has(key))
				return;
			auto *sb = new QSpinBox;
			sb->setRange(lo, hi);
			sb->setValue(value(key).toInt());
			sb->setFixedWidth(110);
			const QString k = key;
			QObject::connect(sb, &QSpinBox::valueChanged,
			                 [touch, k, showCost](int v) {
				setValue(k, QString::number(v));
				showCost();
				touch();
			});
			addRow(c, title, desc, sb);
			refreshers << [sb, k, showCost] {
				QSignalBlocker b(sb);
				sb->setValue(value(k).toInt());
				showCost();
			};
		};

		if (has("section_split")) {
			auto *cb = new Toggle;
			cb->setChecked(truthy(value("section_split")));
			QObject::connect(cb, &QCheckBox::toggled,
			                 [touch, showCost](bool on) {
				setValue("section_split", on ? "true" : "false");
				showCost();
				touch();
			});
			addRow(c, "Split to fit the screen (recommended)",
			       "Chooses the tiles from the screen's shape instead of the "
			       "numbers below: on a widescreen, two full-height square tiles "
			       "that together cover the width, with the overlap worked out so "
			       "nothing is wasted. That is about the detail of a 4x4 grid for "
			       "a fraction of the cost, which leaves the detector fast enough "
			       "to keep up with motion. Turn it off to use the grid below by "
			       "hand. Changing it rebuilds the detector, so the screen "
			       "blanks for a moment.", cb);
			refreshers << [cb, showCost] {
				QSignalBlocker b(cb);
				cb->setChecked(truthy(value("section_split")));
				showCost();
			};
		}

		/* BC_MAX_GRID in preprocess.h is 4; offering more here would be
		 * silently clamped by the core. */
		addInt("section_cols", "Sections across",
		       "How many tiles the screen is divided into left to right. "
		       "1 is the whole screen in one piece.", 1, 4);
		addInt("section_rows", "Sections down",
		       "How many tiles top to bottom.", 1, 4);

		if (has("section_overlap")) {
			auto *sb = new QDoubleSpinBox;
			sb->setRange(0.0, 0.9);
			sb->setSingleStep(0.05);
			sb->setDecimals(2);
			sb->setValue(value("section_overlap").toDouble());
			sb->setFixedWidth(110);
			QObject::connect(sb, &QDoubleSpinBox::valueChanged,
			                 [touch](double v) {
				setValue("section_overlap", QString::number(v, 'f', 2));
				touch();
			});
			addRow(c, "Overlap between sections",
			       "Fraction of a tile shared with its neighbour. Something "
			       "lying across a seam is cut in half by it, and the overlap "
			       "is what gives each half enough context to still be found. "
			       "The two halves are rejoined into one cover afterwards.", sb);
			refreshers << [sb] {
				QSignalBlocker b(sb);
				sb->setValue(value("section_overlap").toDouble());
			};
		}

		if (has("section_full_frame")) {
			auto *cb = new Toggle;
			cb->setChecked(truthy(value("section_full_frame")));
			QObject::connect(cb, &QCheckBox::toggled,
			                 [touch, showCost](bool on) {
				setValue("section_full_frame", on ? "true" : "false");
				showCost();
				touch();
			});
			addRow(c, "Also scan the whole screen",
			       "Runs one extra pass over the entire screen alongside the "
			       "grid. Without it a grid trades large things for small ones: "
			       "something filling the screen is split across every tile, and "
			       "no fragment still looks like the object, so nothing is "
			       "reported at all. Ignored when there is only one section.", cb);
			refreshers << [cb, showCost] {
				QSignalBlocker b(cb);
				cb->setChecked(truthy(value("section_full_frame")));
				showCost();
			};
		}

		showCost();
		cost->setContentsMargins(0, 0, 0, 12);
		c->box->addWidget(cost);
	}
	tDetect->addStretch(1);

	/* --- tab: what to cover ---------------------------------------------- */
	auto *tClasses = newTab("What to cover");

	{
		auto *intro = new QLabel(
			"Tick a body part to cover it whenever it is detected. This list "
			"is the loaded model's own — change the model on the "
			"Censoring tab and it is rebuilt to match.");
		intro->setObjectName("note");
		intro->setWordWrap(true);
		QFont f = intro->font();
		f.setPointSizeF(f.pointSizeF() * 0.92);
		intro->setFont(f);
		tClasses->addWidget(intro);
	}

	auto *bulk = new QHBoxLayout;
	auto *allBtn  = new QPushButton("Cover all");
	auto *noneBtn = new QPushButton("Cover none");
	allBtn->setToolTip("Cover every class this model can detect.");
	noneBtn->setToolTip("Cover nothing. The censor warns when set this way, "
	                    "because a screen covering nothing looks the same as "
	                    "a censor that has failed.");
	bulk->addWidget(allBtn);
	bulk->addWidget(noneBtn);
	bulk->addStretch();
	tClasses->addLayout(bulk);

	/*
	 * Classes from *other* models are in the file too, so that switching back
	 * restores the toggles rather than the defaults. They are not offered
	 * here: they cannot do anything until that model is loaded, and drawn as
	 * plain rows they were indistinguishable from the live ones -- with three
	 * taxonomies seen, two thirds of this panel did nothing. The core files
	 * them under [classes.inactive]; the count is shown so it is clear they
	 * are kept rather than lost.
	 */
	auto *inactiveNote = new QLabel;
	inactiveNote->setObjectName("note");
	inactiveNote->setWordWrap(true);
	{
		QFont f = inactiveNote->font();
		f.setPointSizeF(f.pointSizeF() * 0.92);
		inactiveNote->setFont(f);
	}

	QWidget *gridHost = nullptr;
	QList<QCheckBox *> classBoxes;

	/*
	 * Is this line one of the loaded model's classes?
	 *
	 * Normally the section says so. But the file is meant to be editable by
	 * hand, and a [classes] header deleted by accident would leave every
	 * class unrecognised and the panel empty -- offering to censor nothing,
	 * quietly, which is the failure this program must never present as
	 * normal. So when the file has no [classes] section at all, fall back to
	 * the old rule of "not a global setting, therefore a class". Sections win
	 * whenever they are present; this only covers their absence.
	 */
	auto isClassLine = [](const Line &l, bool sectioned) {   /* nums is static */
		if (l.key.isEmpty())
			return false;
		if (sectioned)
			return l.section == "classes";
		if (l.section.startsWith("classes."))
			return false;
		if (l.key.startsWith("strength_"))
			return false;
		/*
		 * Every non-class key the core writes has to be listed here or
		 * it is drawn as a body part to censor. Kept in step with
		 * config.c: section_* and join_waste were added to the file long
		 * after this list and were missing from it.
		 */
		for (const char *g : { "enabled", "join", "join_waste", "reverse",
		                       "eye_bar", "eye_bar_effect", "eye_bar_face",
		                       "model", "landmark_model",
		                       "seg_model", "seg_margin",
		                       "effect", "caption",
		                       "text_color", "section_cols", "section_rows",
		                       "section_overlap", "section_full_frame",
		                       "outline", "outline_color", "shadow",
		                       "growth", "growth_max", "takeover_ms",
		                       "shadow_opacity", "censor_color" })
			if (l.key == QLatin1String(g))
				return false;
		for (const NumSpec &n : nums)
			if (l.key == QLatin1String(n.key))
				return false;
		return true;
	};

	auto rebuildClasses = [&] {
		delete gridHost;          /* drops its children with it */
		classBoxes.clear();

		bool sectioned = false;
		for (const Line &l : g_lines)
			if (l.section == "classes" && !l.key.isEmpty())
				sectioned = true;

		gridHost = new QWidget;
		auto *stack = new QVBoxLayout(gridHost);
		stack->setContentsMargins(0, 0, 0, 0);
		stack->setSpacing(16);

		for (int g = 0; g < N_GROUPS; g++) {
			QStringList keys;
			for (const Line &l : g_lines)
				if (isClassLine(l, sectioned) &&
				    groupOf(l.key) == g)
					keys << l.key;
			if (keys.isEmpty())
				continue;

			Card *c = makeCard(stack, QLatin1String(GROUPS[g]));

			/*
			 * Name-and-switch, in columns. Not addRow(): these have
			 * no description to wrap, and one per line would make
			 * twenty classes a page and a half of scrolling.
			 */
			auto *host = new ClassGrid;
			for (const QString &k : keys) {
				auto *name = new QLabel(prettyName(k));
				/* The model's own name for it, for anyone
				 * matching this against the config file. */
				name->setToolTip(k);

				/*
				 * A class line is a boolean or the name of the
				 * effect covering that class, so the switch and
				 * the effect share one value and have to be
				 * written together. They travel as one cell:
				 * ClassGrid lays out a label and one widget, and
				 * splitting them into two columns would put the
				 * switch and its effect in different halves of
				 * the grid.
				 */
				/*
				 * The line may carry an `@score` suffix pinning
				 * this class's own detection floor. Split it off
				 * before anything else looks at the value: the
				 * effect match below is an exact compare, so
				 * "fill@0.15" would read as an unrecognised
				 * effect and silently unpin the class.
				 */
				const QString raw = value(k);
				const int atPos = raw.indexOf('@');
				const QString cur = (atPos >= 0)
				                  ? raw.left(atPos).trimmed()
				                  : raw;
				double curConf = 0.0;
				if (atPos >= 0) {
					bool okc = false;
					const double s =
						raw.mid(atPos + 1).trimmed()
						   .toDouble(&okc);
					if (okc && s > 0.0 && s < 1.0)
						curConf = s;
				}

				int pinned = -1;
				for (int e = 0; e < (int)(sizeof EFFECTS /
				                          sizeof EFFECTS[0]); e++)
					if (!cur.compare(QLatin1String(EFFECTS[e].key),
					                 Qt::CaseInsensitive)) {
						pinned = e;
						break;
					}

				auto *cell = new QWidget;
				auto *cl = new QHBoxLayout(cell);
				cl->setContentsMargins(0, 0, 0, 0);
				cl->setSpacing(8);

				auto *eff = new QComboBox;
				/* Index 0 is "no opinion" -- follow whatever the
				 * Effect tab says. Not a copy of the current
				 * global effect: picking one here has to mean
				 * "pin", or changing the global effect later
				 * would move nothing. */
				eff->addItem("Default");
				for (const EffectInfo &e : EFFECTS)
					eff->addItem(QLatin1String(e.key));
				eff->setCurrentIndex(pinned + 1);
				eff->setToolTip("Effect covering this class. "
				                "Default follows the Effect tab.");

				/*
				 * This class's own detection floor. Zero is the
				 * "auto" position rather than a real threshold:
				 * a floor of zero would admit every channel the
				 * model emitted at any score, so the one value
				 * that must not be settable here is the one at
				 * the bottom of the range.
				 */
				auto *conf = new QDoubleSpinBox;
				conf->setRange(0.0, 0.95);
				conf->setSingleStep(0.05);
				conf->setDecimals(2);
				conf->setSpecialValueText("auto");
				conf->setValue(curConf);
				conf->setFixedWidth(72);
				conf->setToolTip(
					"Score this class must reach before it is "
					"covered.\n\"auto\" follows the "
					"Detection tab's confidence.");

				auto *cb = new Toggle;
				cb->setToolTip(k);
				cb->setChecked(pinned >= 0 || truthy(cur));
				/* An effect or a threshold on a class that is
				 * off would be a setting with no visible
				 * result. */
				eff->setEnabled(cb->isChecked());
				conf->setEnabled(cb->isChecked());

				auto write = [touch, k, cb, eff, conf] {
					QString v;
					if (!cb->isChecked())
						v = "false";
					else if (eff->currentIndex() <= 0)
						v = "true";
					else
						v = eff->currentText();
					/*
					 * The suffix is written only when there
					 * is one to write. Emitting the global
					 * value onto every class would freeze
					 * them all at today's number, after
					 * which moving the global slider would
					 * move nothing.
					 */
					if (cb->isChecked() && conf->value() > 0.0)
						v += QString("@%1").arg(
							conf->value(), 0, 'f', 2);
					setValue(k, v);
					touch();
				};
				QObject::connect(cb, &QCheckBox::toggled,
				                 [eff, conf, write](bool on) {
					eff->setEnabled(on);
					conf->setEnabled(on);
					write();
				});
				QObject::connect(eff, &QComboBox::currentTextChanged,
				                 [write](const QString &) { write(); });
				QObject::connect(conf, &QDoubleSpinBox::valueChanged,
				                 [write](double) { write(); });

				cl->addWidget(eff);
				cl->addWidget(conf);
				cl->addWidget(cb);
				host->addEntry(name, cell);
				classBoxes.append(cb);
			}
			c->box->addWidget(host);
		}

		if (classBoxes.isEmpty())
			stack->addWidget(new QLabel(
				"The censor has not written a class list yet."));

		int inactive = 0;
		for (const Line &l : g_lines)
			if (!l.key.isEmpty() && l.section.startsWith("classes."))
				inactive++;
		inactiveNote->setText(inactive
			? QString("%1 more setting(s) belong to models that are not "
			          "loaded. They are kept, and come back when you "
			          "switch to that model.").arg(inactive)
			: QString());
		inactiveNote->setVisible(inactive > 0);

		/* Above the note and the trailing stretch. */
		tClasses->insertWidget(tClasses->count() - 2, gridHost);
	};

	tClasses->addWidget(inactiveNote);
	tClasses->addStretch(1);
	rebuildClasses();

	auto setAll = [&classBoxes](bool on) {
		for (QCheckBox *cb : classBoxes)
			cb->setChecked(on);   /* each fires toggled; save is debounced */
	};
	QObject::connect(allBtn,  &QPushButton::clicked, [setAll] { setAll(true); });
	QObject::connect(noneBtn, &QPushButton::clicked, [setAll] { setAll(false); });

	/* --- status --------------------------------------------------------- */
	{
		auto *sep = new QFrame;
		sep->setObjectName("sep");
		sep->setFixedHeight(1);
		root->addWidget(sep);

		auto *bar = new QHBoxLayout;
		bar->setContentsMargins(18, 8, 18, 10);
		bar->addWidget(status);
		root->addLayout(bar);
	}

	/*
	 * Follow the file if something else changes it -- the core rewrites it
	 * on first run, and it is meant to be hand-editable. Without this the
	 * GUI would hold a stale copy and quietly stamp it back over their edit
	 * on the next click.
	 */
	auto *watcher = new QFileSystemWatcher(&win);
	watcher->addPath(g_path);

	/*
	 * Re-arm. Watching a *file* means watching an inode, and every writer
	 * here replaces the file rather than editing it in place -- we do, the
	 * core does, and so do most editors -- so the watch is dropped on every
	 * single write and has to be put back.
	 *
	 * It must happen before any early return. It used to sit at the end of
	 * the handler, after a guard that returned when the write was our own,
	 * which meant the GUI's own save permanently disarmed its watch: the
	 * handler bailed before reaching the re-arm and no later change was ever
	 * seen. That is why switching model needed a restart -- the GUI writes
	 * `model`, disarms itself, and then never hears the core rewrite
	 * [classes] with the new taxonomy a moment later.
	 */
	auto rearm = [watcher] {
		if (!watcher->files().contains(g_path))
			watcher->addPath(g_path);
	};

	auto reloadIfChanged = [status, &refreshers, &rebuildClasses] {
		if (!diskDiffers())
			return;
		if (!load(g_path))
			return;
		for (const auto &r : refreshers)
			r();
		/*
		 * Rebuild rather than re-tick. A model swap rewrites the file with
		 * a different taxonomy, and only the values were being refreshed
		 * before -- so after swapping, the panel went on showing the
		 * previous model's classes, each one wired to a key the running
		 * detector no longer had.
		 */
		rebuildClasses();
		status->setText("Reloaded (file changed on disk).");
	};

	QObject::connect(watcher, &QFileSystemWatcher::fileChanged,
	                 [rearm, reloadIfChanged] {
		rearm();
		reloadIfChanged();
	});

	/*
	 * A backstop on the same principle the core uses: poll the mtime. inotify
	 * on a path that keeps being replaced has too many ways to go quiet for
	 * the panel's correctness to rest on it alone -- and the cost of being
	 * wrong is a settings window describing a model that is not loaded, which
	 * is worse than a stat() four times a second. The core polls this same
	 * file at 4 Hz for exactly this reason.
	 */
	auto *poll = new QTimer(&win);
	poll->setInterval(250);
	QObject::connect(poll, &QTimer::timeout, [rearm, reloadIfChanged] {
		rearm();
		reloadIfChanged();
	});
	poll->start();

	/*
	 * Run every refresher once, so the panel starts in the state the file
	 * describes rather than the state each control happened to be built in.
	 * Most set themselves from the file as they are created, but the ones
	 * whose appearance depends on the *kind* of value -- the colour swatch,
	 * which has no meaning unless a colour was chosen -- had no initial pass
	 * at all and started showing an empty box. They all block signals, so
	 * this cannot write anything back.
	 */
	for (const auto &r : refreshers)
		r();

	status->setText(QString("Editing %1").arg(QFileInfo(g_path).fileName()));
	win.setMinimumSize(400, 380);
	win.resize(680, 760);
	win.show();

	/*
	 * BC_GUI_SHOT=<path>: render the window to a PNG and exit.
	 *
	 * DEBUG ONLY, but kept because the alternative is worse. This panel's
	 * failures are layout failures -- text clipped mid-sentence, a section
	 * squeezed to a sliver -- and none of them show up in the source or in
	 * any runtime check. A compositor screenshot cannot be scripted reliably
	 * (it grabs whatever happens to be focused), so grabbing the widget
	 * itself is the only way to see what was actually drawn.
	 *
	 * BC_GUI_SHOT_SIZE=WxH forces a window size first, and BC_GUI_SHOT_TAB
	 * picks the tab, which is how "does it survive a small window" gets
	 * tested without a human dragging an edge.
	 */
	const QByteArray shot = qgetenv("BC_GUI_SHOT");
	if (!shot.isEmpty()) {
		const QByteArray sz = qgetenv("BC_GUI_SHOT_SIZE");
		if (!sz.isEmpty()) {
			const QList<QByteArray> wh = sz.split('x');
			if (wh.size() == 2)
				win.resize(wh[0].toInt(), wh[1].toInt());
		}
		tabs->setCurrentIndex(qgetenv("BC_GUI_SHOT_TAB").toInt());
		const QString path = QString::fromLocal8Bit(shot);
		QTimer::singleShot(700, [&win, path] {
			win.grab().save(path);
			QApplication::quit();
		});
	}
	return app.exec();
}

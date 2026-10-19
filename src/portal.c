#include "portal.h"
#include "util.h"

#include "sdbus.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

#define PORTAL_DEST  "org.freedesktop.portal.Desktop"
#define PORTAL_PATH  "/org/freedesktop/portal/desktop"
#define SCREENCAST   "org.freedesktop.portal.ScreenCast"
#define REQUEST_IF   "org.freedesktop.portal.Request"

/* SelectSources source types */
#define SRC_MONITOR  1u

/* cursor modes */
#define CURSOR_HIDDEN   1u
#define CURSOR_EMBEDDED 2u

/* persist_mode: 2 = persist until explicitly revoked */
#define PERSIST_UNTIL_REVOKED 2u

/* ------------------------------------------------------------------ */
/* Request/Response plumbing                                          */
/* ------------------------------------------------------------------ */

typedef int (*results_fn)(sd_bus_message *m, void *user);

struct req_ctx {
	int        done;
	uint32_t   response;   /* 0 ok, 1 user cancelled, 2 error */
	results_fn parse;
	void      *user;
	int        rc;
};

static int on_response(sd_bus_message *m, void *userdata, sd_bus_error *e)
{
	struct req_ctx *c = userdata;
	uint32_t resp = 2;

	c->rc = sd_bus_message_read(m, "u", &resp);
	if (c->rc >= 0) {
		c->response = resp;
		if (resp == 0 && c->parse)
			c->rc = c->parse(m, c->user);
	}
	c->done = 1;
	return 0;
}

/* Walk an a{sv}, handing each (key, variant-positioned message) to cb. */
static int dict_foreach(sd_bus_message *m,
                        int (*cb)(const char *key, sd_bus_message *v, void *u),
                        void *u)
{
	int r = sd_bus_message_enter_container(m, SD_BUS_TYPE_ARRAY, "{sv}");
	if (r < 0)
		return r;

	for (;;) {
		r = sd_bus_message_enter_container(m, SD_BUS_TYPE_DICT_ENTRY, "sv");
		if (r <= 0)
			break;

		const char *key = NULL;
		r = sd_bus_message_read(m, "s", &key);
		if (r < 0)
			return r;

		char type = 0;
		const char *contents = NULL;
		r = sd_bus_message_peek_type(m, &type, &contents);
		if (r < 0)
			return r;

		r = sd_bus_message_enter_container(m, SD_BUS_TYPE_VARIANT, contents);
		if (r < 0)
			return r;

		r = cb(key, m, u);
		if (r < 0)
			return r;

		sd_bus_message_skip(m, NULL);
		sd_bus_message_exit_container(m); /* variant */
		sd_bus_message_exit_container(m); /* dict entry */
	}

	sd_bus_message_exit_container(m); /* array */
	return 0;
}

/* ":1.234" -> "1_234", as the portal spells it in Request object paths. */
static char *sender_fragment(sd_bus *bus)
{
	const char *unique = NULL;
	if (sd_bus_get_unique_name(bus, &unique) < 0 || !unique)
		return NULL;
	if (*unique == ':')
		unique++;

	char *s = strdup(unique);
	if (!s)
		return NULL;
	for (char *p = s; *p; p++)
		if (*p == '.')
			*p = '_';
	return s;
}

static void make_token(char *buf, size_t n, const char *prefix)
{
	static unsigned counter;
	snprintf(buf, n, "bc_%s_%u_%d", prefix, counter++, (int)getpid());
}

/*
 * Install the Response match *before* issuing the call, then pump the bus
 * until it fires. The portal replies with the Request path, but racing the
 * signal against the method reply is a real hazard, so we predict the path.
 */
static int await_request(sd_bus *bus, const char *req_path,
                         struct req_ctx *ctx, const char *what)
{
	for (;;) {
		int r = sd_bus_process(bus, NULL);
		if (r < 0) {
			ERR("%s: sd_bus_process: %s", what, strerror(-r));
			return r;
		}
		if (ctx->done)
			break;
		if (r > 0)
			continue;

		r = sd_bus_wait(bus, (uint64_t)-1);
		if (r < 0) {
			ERR("%s: sd_bus_wait: %s", what, strerror(-r));
			return r;
		}
	}

	if (ctx->rc < 0) {
		ERR("%s: malformed response: %s", what, strerror(-ctx->rc));
		return ctx->rc;
	}
	if (ctx->response == 1) {
		ERR("%s: cancelled by user", what);
		return -ECANCELED;
	}
	if (ctx->response != 0) {
		ERR("%s: portal returned %u", what, ctx->response);
		return -EIO;
	}
	return 0;
}

/* ------------------------------------------------------------------ */
/* Result parsers                                                     */
/* ------------------------------------------------------------------ */

static int cb_session_handle(const char *key, sd_bus_message *v, void *u)
{
	char **out = u;
	if (strcmp(key, "session_handle") == 0) {
		const char *s = NULL;
		int r = sd_bus_message_read(v, "s", &s);
		if (r < 0)
			return r;
		free(*out);
		*out = strdup(s);
	}
	return 0;
}

static int parse_session(sd_bus_message *m, void *u)
{
	return dict_foreach(m, cb_session_handle, u);
}

struct start_out {
	uint32_t node_id;
	int      have_node;
	char    *restore_token;
};

static int cb_start(const char *key, sd_bus_message *v, void *u)
{
	struct start_out *o = u;
	int r;

	if (strcmp(key, "restore_token") == 0) {
		const char *s = NULL;
		r = sd_bus_message_read(v, "s", &s);
		if (r < 0)
			return r;
		free(o->restore_token);
		o->restore_token = strdup(s);
		return 0;
	}

	if (strcmp(key, "streams") != 0)
		return 0;

	/* streams: a(ua{sv}) — take the first, we only ever ask for one. */
	r = sd_bus_message_enter_container(v, SD_BUS_TYPE_ARRAY, "(ua{sv})");
	if (r < 0)
		return r;

	while ((r = sd_bus_message_enter_container(v, SD_BUS_TYPE_STRUCT,
	                                           "ua{sv}")) > 0) {
		uint32_t node = 0;
		r = sd_bus_message_read(v, "u", &node);
		if (r < 0)
			return r;

		if (!o->have_node) {
			o->node_id = node;
			o->have_node = 1;
		}

		sd_bus_message_skip(v, "a{sv}");
		sd_bus_message_exit_container(v);
	}

	sd_bus_message_exit_container(v);
	return 0;
}

static int parse_start(sd_bus_message *m, void *u)
{
	return dict_foreach(m, cb_start, u);
}

/* ------------------------------------------------------------------ */
/* restore token cache                                                */
/* ------------------------------------------------------------------ */

static char *restore_load(const char *path)
{
	if (!path)
		return NULL;
	FILE *f = fopen(path, "r");
	if (!f)
		return NULL;

	char buf[1024];
	if (!fgets(buf, sizeof buf, f)) {
		fclose(f);
		return NULL;
	}
	fclose(f);

	buf[strcspn(buf, "\r\n")] = 0;
	return buf[0] ? strdup(buf) : NULL;
}

static void restore_save(const char *path, const char *token)
{
	if (!path || !token || !*token)
		return;
	FILE *f = fopen(path, "w");
	if (!f) {
		WARN("could not write restore token to %s: %s",
		     path, strerror(errno));
		return;
	}
	fprintf(f, "%s\n", token);
	fclose(f);
	LOG("cached restore token -> %s (next run should not prompt)", path);
}

/* ------------------------------------------------------------------ */

int bc_portal_acquire(struct bc_portal *out, const char *restore_path)
{
	sd_bus            *bus     = NULL;
	sd_bus_error       err     = SD_BUS_ERROR_NULL;
	sd_bus_message    *reply   = NULL;
	sd_bus_message    *call    = NULL;
	sd_bus_slot       *slot    = NULL;
	char              *sender  = NULL;
	char              *session = NULL;
	char              *restore = NULL;
	struct start_out   start   = { 0 };
	int                rc;

	out->pipewire_fd = -1;
	out->node_id     = 0;

	rc = sd_bus_open_user(&bus);
	if (rc < 0) {
		ERR("cannot connect to session bus: %s", strerror(-rc));
		return rc;
	}

	sender = sender_fragment(bus);
	if (!sender) {
		rc = -ENOMEM;
		goto done;
	}

	/* --- which cursor modes does this portal actually offer? --- */
	uint32_t cursor_modes = 0;
	if (sd_bus_get_property_trivial(bus, PORTAL_DEST, PORTAL_PATH,
	                                SCREENCAST, "AvailableCursorModes",
	                                &err, 'u', &cursor_modes) < 0) {
		WARN("AvailableCursorModes unreadable (%s); assuming hidden",
		     err.message ? err.message : "?");
		sd_bus_error_free(&err);
		cursor_modes = CURSOR_HIDDEN;
	}
	uint32_t cursor = (cursor_modes & CURSOR_EMBEDDED) ? CURSOR_EMBEDDED
	                                                   : CURSOR_HIDDEN;

	/* ---------------- CreateSession ---------------- */
	{
		char tok[64], stok[64], path[512];
		make_token(tok, sizeof tok, "cs");
		make_token(stok, sizeof stok, "sess");
		snprintf(path, sizeof path,
		         "%s/request/%s/%s", PORTAL_PATH, sender, tok);

		struct req_ctx ctx = { .parse = parse_session, .user = &session };
		rc = sd_bus_match_signal(bus, &slot, PORTAL_DEST, path,
		                         REQUEST_IF, "Response", on_response, &ctx);
		if (rc < 0)
			goto done;

		rc = sd_bus_message_new_method_call(bus, &call, PORTAL_DEST,
		                                    PORTAL_PATH, SCREENCAST,
		                                    "CreateSession");
		if (rc < 0) goto done;
		rc = sd_bus_message_open_container(call, 'a', "{sv}");
		if (rc < 0) goto done;
		sd_bus_message_append(call, "{sv}", "handle_token", "s", tok);
		sd_bus_message_append(call, "{sv}", "session_handle_token", "s", stok);
		rc = sd_bus_message_close_container(call);
		if (rc < 0) goto done;

		rc = sd_bus_call(bus, call, 0, &err, &reply);
		if (rc < 0) {
			ERR("CreateSession: %s", err.message ? err.message : strerror(-rc));
			goto done;
		}
		call  = sd_bus_message_unref(call);
		reply = sd_bus_message_unref(reply);

		rc = await_request(bus, path, &ctx, "CreateSession");
		slot = sd_bus_slot_unref(slot);
		if (rc < 0)
			goto done;

		if (!session) {
			ERR("CreateSession returned no session_handle");
			rc = -EIO;
			goto done;
		}
		LOG("session: %s", session);
	}

	/* ---------------- SelectSources ---------------- */
	restore = restore_load(restore_path);
	if (restore)
		LOG("reusing cached restore token (should skip the dialog)");

	{
		char tok[64], path[512];
		make_token(tok, sizeof tok, "ss");
		snprintf(path, sizeof path,
		         "%s/request/%s/%s", PORTAL_PATH, sender, tok);

		struct req_ctx ctx = { 0 };
		rc = sd_bus_match_signal(bus, &slot, PORTAL_DEST, path,
		                         REQUEST_IF, "Response", on_response, &ctx);
		if (rc < 0)
			goto done;

		rc = sd_bus_message_new_method_call(bus, &call, PORTAL_DEST,
		                                    PORTAL_PATH, SCREENCAST,
		                                    "SelectSources");
		if (rc < 0) goto done;
		rc = sd_bus_message_append(call, "o", session);
		if (rc < 0) goto done;
		rc = sd_bus_message_open_container(call, 'a', "{sv}");
		if (rc < 0) goto done;
		sd_bus_message_append(call, "{sv}", "handle_token", "s", tok);
		sd_bus_message_append(call, "{sv}", "types", "u", SRC_MONITOR);
		sd_bus_message_append(call, "{sv}", "multiple", "b", 0);
		sd_bus_message_append(call, "{sv}", "cursor_mode", "u", cursor);
		sd_bus_message_append(call, "{sv}", "persist_mode", "u",
		                      PERSIST_UNTIL_REVOKED);
		if (restore)
			sd_bus_message_append(call, "{sv}", "restore_token", "s", restore);
		rc = sd_bus_message_close_container(call);
		if (rc < 0) goto done;

		rc = sd_bus_call(bus, call, 0, &err, &reply);
		if (rc < 0) {
			ERR("SelectSources: %s", err.message ? err.message : strerror(-rc));
			goto done;
		}
		call  = sd_bus_message_unref(call);
		reply = sd_bus_message_unref(reply);

		rc = await_request(bus, path, &ctx, "SelectSources");
		slot = sd_bus_slot_unref(slot);
		if (rc < 0)
			goto done;
	}

	/* ---------------- Start ---------------- */
	{
		char tok[64], path[512];
		make_token(tok, sizeof tok, "st");
		snprintf(path, sizeof path,
		         "%s/request/%s/%s", PORTAL_PATH, sender, tok);

		struct req_ctx ctx = { .parse = parse_start, .user = &start };
		rc = sd_bus_match_signal(bus, &slot, PORTAL_DEST, path,
		                         REQUEST_IF, "Response", on_response, &ctx);
		if (rc < 0)
			goto done;

		rc = sd_bus_message_new_method_call(bus, &call, PORTAL_DEST,
		                                    PORTAL_PATH, SCREENCAST, "Start");
		if (rc < 0) goto done;
		rc = sd_bus_message_append(call, "os", session, "");
		if (rc < 0) goto done;
		rc = sd_bus_message_open_container(call, 'a', "{sv}");
		if (rc < 0) goto done;
		sd_bus_message_append(call, "{sv}", "handle_token", "s", tok);
		rc = sd_bus_message_close_container(call);
		if (rc < 0) goto done;

		LOG("waiting for consent dialog (pick the monitor to censor)...");
		rc = sd_bus_call(bus, call, 0, &err, &reply);
		if (rc < 0) {
			ERR("Start: %s", err.message ? err.message : strerror(-rc));
			goto done;
		}
		call  = sd_bus_message_unref(call);
		reply = sd_bus_message_unref(reply);

		rc = await_request(bus, path, &ctx, "Start");
		slot = sd_bus_slot_unref(slot);
		if (rc < 0)
			goto done;

		if (!start.have_node) {
			ERR("Start returned no stream");
			rc = -EIO;
			goto done;
		}
		LOG("stream node id: %u", start.node_id);
		restore_save(restore_path, start.restore_token);
	}

	/* ---------------- OpenPipeWireRemote ---------------- */
	{
		rc = sd_bus_message_new_method_call(bus, &call, PORTAL_DEST,
		                                    PORTAL_PATH, SCREENCAST,
		                                    "OpenPipeWireRemote");
		if (rc < 0) goto done;
		rc = sd_bus_message_append(call, "o", session);
		if (rc < 0) goto done;
		rc = sd_bus_message_open_container(call, 'a', "{sv}");
		if (rc < 0) goto done;
		rc = sd_bus_message_close_container(call);
		if (rc < 0) goto done;

		rc = sd_bus_call(bus, call, 0, &err, &reply);
		if (rc < 0) {
			ERR("OpenPipeWireRemote: %s",
			    err.message ? err.message : strerror(-rc));
			goto done;
		}

		int fd = -1;
		rc = sd_bus_message_read(reply, "h", &fd);
		if (rc < 0 || fd < 0) {
			ERR("OpenPipeWireRemote returned no fd");
			rc = rc < 0 ? rc : -EIO;
			goto done;
		}

		/* sd-bus owns the fd with the message; keep our own copy. */
		out->pipewire_fd = dup(fd);
		if (out->pipewire_fd < 0) {
			ERR("dup(pipewire fd): %s", strerror(errno));
			rc = -errno;
			goto done;
		}
		out->node_id = start.node_id;
		rc = 0;
	}

done:
	sd_bus_error_free(&err);
	sd_bus_message_unref(call);
	sd_bus_message_unref(reply);
	sd_bus_slot_unref(slot);
	free(sender);
	free(session);
	free(restore);
	free(start.restore_token);
	/*
	 * Deliberately leaking the bus connection: closing it drops the portal
	 * session and kills the stream. Phase 1 exits soon after; a long-running
	 * build must hold this open for the process lifetime.
	 */
	if (rc < 0)
		sd_bus_unref(bus);
	return rc;
}

void bc_portal_release(struct bc_portal *p)
{
	if (p->pipewire_fd >= 0) {
		close(p->pipewire_fd);
		p->pipewire_fd = -1;
	}
}

#ifndef BC_PORTAL_H
#define BC_PORTAL_H

#include <stdint.h>

/*
 * xdg-desktop-portal ScreenCast negotiation.
 *
 * Produces the two things the PipeWire side needs: a file descriptor for the
 * PipeWire remote, and the node id of the stream to connect to.
 *
 * DESIGN.md §3[1]: we ask for persist_mode=2 and cache the returned
 * restore_token, so the user picks their monitor once rather than on every
 * launch. Repeated consent prompts are what make these tools unusable on
 * Wayland.
 */

struct bc_portal {
	int      pipewire_fd;   /* fd for pw_context_connect_fd(), -1 if none  */
	uint32_t node_id;       /* PipeWire node to connect the stream to      */
};

/*
 * Runs the full CreateSession → SelectSources → Start → OpenPipeWireRemote
 * dance. Blocks while the user is in the consent dialog.
 *
 * restore_path: file used to cache the restore token across runs (may be NULL).
 * Returns 0 on success, negative errno-ish on failure.
 */
int bc_portal_acquire(struct bc_portal *out, const char *restore_path);

void bc_portal_release(struct bc_portal *p);

#endif /* BC_PORTAL_H */

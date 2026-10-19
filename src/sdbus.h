/*
 * sd-bus, from whichever library on this machine provides it.
 *
 * The ScreenCast portal is plain D-Bus and sd-bus is the smallest way to speak
 * it, but the library carrying that API differs by distro: systemd everywhere
 * mainstream, basu on systemd-free distros (Void, Artix, Chimera), elogind on
 * some Gentoo and Devuan setups. The API is the same in all three -- only the
 * header path differs -- so the choice belongs in the build, not in the code
 * that calls it. The Makefile probes for one and defines the matching macro.
 *
 * Deliberately a header rather than a compat layer: there is nothing to
 * abstract here, and a shim that wrapped sd_bus_* calls would be a second
 * thing to keep in sync with a library that is already stable.
 */
#ifndef BC_SDBUS_H
#define BC_SDBUS_H

#if defined(BC_SDBUS_BASU)
#  include <basu/sd-bus.h>
#elif defined(BC_SDBUS_ELOGIND)
#  include <elogind/sd-bus.h>
#else
#  include <systemd/sd-bus.h>
#endif

#endif /* BC_SDBUS_H */

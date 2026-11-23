"""Import guard for the optional plotting dependency."""


def eyepiece():
    """Return eyepiece or raise with the install hint for this extra.

    Returns:
        The imported ``eyepiece`` module.
    """
    try:
        import eyepiece as ep
    except ImportError as err:
        msg = "yippy.sinc.viz requires eyepiece: pip install 'yippy[viz]'"
        raise ImportError(msg) from err
    return ep

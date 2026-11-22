"""
tga2tpc-py - convert between TGA textures and BioWare TPC textures.

This is a Python reimplementation of ndix UR's tga2tpc utility.  It reads Targa (.tga) images, optionally uses a sibling/explict .txi file, generates mipmaps and writes BioWare TPC textures (DXT1, DXT3, DXT5, RGB, RGBA or greyscale).  The companion ``xoreostex2tga`` / ``tex2tga`` command converts TPC, DDS, TXB and TGA textures back to TGA.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]

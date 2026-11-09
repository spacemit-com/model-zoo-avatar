"""Two-model SDK. Native backends are loaded lazily; no model downloads on import."""
from .pipeline import FeatherTalkService, RenderResult

_NATIVE={'RuntimeOptions','RenderOptions','FeatureBatch','FeatherHubertExtractor','UnetLipGenerator','PreparedFrame','runtime_version'}
def __getattr__(name):
    if name=='__version__':
        from importlib.metadata import version,PackageNotFoundError
        try:return version('spacemit-feathertalk')
        except PackageNotFoundError:
            from pathlib import Path
            source=Path(__file__).resolve().parents[2]/'VERSION'
            return source.read_text().strip() if source.exists() else '0+unknown'
    if name in _NATIVE:
        from . import _feathertalk
        return getattr(_feathertalk,name)
    raise AttributeError(name)

__all__=['FeatherTalkService','RenderResult',*sorted(_NATIVE)]

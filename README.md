# z-vsPyScripts

Z's weird vapoursynth python script(s)

## Usage

### zvs

`import zvs`

### .env.zvs(.global)

NOT necessary. Will fall back to default values if it can't be found.  
The global one resides alongside the main script file zvs.py, and a local one can override global values, you should place it somewhere can be found by python-dotenv, like your project root folder.(that means follow the python-dotenv documents there, yes I'm lazy)

### vsenv

\*cringe\*

```
from vsenv import *
core.num_threads=nn
core.max_cache_size=xxx #change these values can work inside vsenv.py but like if you preview in vsedit it only works for first time so not very well... but other than that i think it's viable
```

\*cringe\*

## Requirements (for zvs)

### python packages

-   [mvsfunc](https://github.com/HomeOfVapourSynthEvolution/mvsfunc)
-   [numpy](https://pypi.org/project/numpy/)(optional)
-   [opencv-python](https://pypi.org/project/opencv-python/)(optional)
-   [PyWavelets](https://pypi.org/project/PyWavelets/)(optional)
-   [scipy](https://pypi.org/project/scipy/)(optional)
-   [python-dotenv](https://pypi.org/project/python-dotenv/)

### python scripts

-   [muvsfunc](https://github.com/WolframRhodium/muvsfunc)
-   [nnedi3_resample](https://github.com/HomeOfVapourSynthEvolution/nnedi3_resample)
-   [xvs](https://github.com/xyx98/my-vapoursynth-script)
-   [finesharp](https://gist.github.com/4re/8676fd350d4b5b223ab9)(optional)

### plugins (most of which should be optional when not using related function and this list is probably incomplete)

-   [BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)
-   [BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) (only VAggregate is used for not-v2 bm3dcuda)
-   *[FFTW3](http://www.fftw.org/install/windows.html)* (dependency of some plugins such as the one above)
-   [mvtools](https://github.com/dubhater/vapoursynth-mvtools) (for degrain4-6 and some speed up use [my build](https://github.com/Mr-Z-2697/vapoursynth-mvtools/releases) before they make new release)
-   [descale](https://github.com/Irrational-Encoding-Wizardry/descale)
-   [nnedi3cl](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL)/[znedi3](https://github.com/sekrit-twc/znedi3)/[nnedi3](https://github.com/dubhater/vapoursynth-nnedi3)
-   [fmtconv](https://gitlab.com/EleonoreMizo/fmtconv)
-   [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)/[f3kdb](https://github.com/SAPikachu/flash3kyuu_deband)
-   [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny)
-   [akarin](https://github.com/AkarinVS/vapoursynth-plugin)
-   [edgefixer](https://github.com/sekrit-twc/EdgeFixer)
-   [KNLMeansCL](https://github.com/pinterf/KNLMeansCL)
-   [nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda)
-   [bilateral](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral)
-   [cas](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS)
-   [vs-removegrain](https://github.com/vapoursynth/vs-removegrain)
-   [ctmf](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)
-   [DualSynth-madVR](https://github.com/Jaded-Encoding-Thaumaturgy/DualSynth-madVR) (and madVR itself of course)

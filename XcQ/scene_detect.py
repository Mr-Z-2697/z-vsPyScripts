'''https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/4c516ceedcac725bc5ded38406f492463494d94c/src/scene_detect.py
BSD 3-Clause License

Copyright (c) 2022, styler00dollar aka sudo rm -rf / --no-preserve-root
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.'''

import numpy as np
import vapoursynth as vs
import functools
import onnxruntime as ort
core = vs.core

# somehow "+rife" models crash mpv but not vapoursynth filter for directshow, happens to me at least, so please use other models when use in mpv.
def scene_detect(
    clip: vs.VideoNode,
    onnx_path: str = r"D:\Misc\scd-models\sc_efficientformerv2_s0+rife46_84119_224_6chIn_softmaxOut_fp16_op17_sim.onnx",
    thresh: float = 0.98,
    fp16: bool = True,
    onnx_res: int = 224,
    ort_provider='Dml',
    resizer=None,
) -> vs.VideoNode:

    sess = ort.InferenceSession(
        onnx_path,
        providers=[f"{ort_provider}ExecutionProvider"],
    )

    def execute(n,f):
        fout=f[0].copy()
        I0 = frame_to_tensor(f[1])
        I1 = frame_to_tensor(f[2])
        I0 = np.expand_dims(I0, 0)
        I1 = np.expand_dims(I1, 0)
        in_sess = np.concatenate([I0, I1], axis=1)
        result = sess.run(None, {"input": in_sess})[0]
        if result > thresh:
            fout.props._SceneChangeNext=1
        else:
            fout.props._SceneChangeNext=0
        fout.props._SceneChangeMetrics=float(result)
        return fout
    clip_down=prepare_clip(clip,onnx_res=onnx_res,fp16=fp16,resizer=resizer)
    return core.std.ModifyFrame(clip,(clip,clip_down,clip_down[1:]),execute)


def frame_to_tensor(frame: vs.VideoFrame):
    return np.stack(
        [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
    )
def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f
def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip, width=image.shape[-1], height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image),
    )
def prepare_clip(clip,onnx_res,fp16,resizer):
    col_fam=clip.format.color_family
    target_fmt=[vs.RGBS,vs.RGBH][fp16]
    resizer=functools.partial(core.resize.Bicubic,filter_param_a=-0.3,filter_param_b=0.15) if resizer is None else resizer
    resizer=functools.partial(resizer,format=target_fmt)
    if clip.format.id==target_fmt and clip.width==clip.height==onnx_res:
        return clip
    if col_fam==vs.RGB:
        return resizer(clip,onnx_res,onnx_res)
    else:
        matrix=clip.get_frame(0).props.get("_Matrix")
        if matrix is None or matrix==2:
            if clip.width>=1280 or clip.height>=720:
                matrix=1
            else:
                matrix=5
        return resizer(clip,onnx_res,onnx_res,matrix_in=matrix)

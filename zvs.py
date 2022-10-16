import os,sys
import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import xvs
import mvsfunc as mvf
import finesharp
import muvsfunc as muf
from functools import partial
from typing import Optional
import nnedi3_resample as nnrs

try:
    from zvs_defaults import *
except:
    nnrs_mode_default='nnedi3'
    bm3d_mode_default='cpu'

nnrs.nnedi3_resample=partial(nnrs.nnedi3_resample,mode=nnrs_mode_default,nns=3,nsize=3,qual=2,pscrn=1)
Nnrs=nnrs

'''
functions:
- pqdenoise
- zmde
- xdbcas
- arop
- pfinesharp (rpfilter (rpclip))
- w2xaa
- knl4a
- wtfmask
- bordermask
- bm3d (copy-paste!)
- n3pv
- rescale, rescalef, multirescale (copy-paste!)
- quack
- bilateraluv
'''

#denoise pq hdr content by partially convert it to bt709 then take the difference back to pq, may yield a better result
def pqdenoise(src,sigma=[1,1,1],lumaonly=False,block_step=7,radius=1,finalest=False,bm3dtyp=bm3d_mode_default,mdegrain=True,tr=2,pel=1,blksize=16,overlap=None,chromamv=True,thsad=100,thsadc=None,thscd1=400,thscd2=130,nl=100,contrasharp=1,to709=1,show='output',limit=255,limitc=None,sigma2=None,radius2=None):
    if lumaonly:
        chromamv=False
        chromaclip=src
        src=xvs.getY(src)

    src=src.fmtc.bitdepth(bits=16)
    denoised=sdr=core.resize.Bicubic(src,transfer_in=16,transfer=1,nominal_luminance=nl) if to709 else src
    if mdegrain:
        limitc=limitc if limitc else limit
        denoised=zmde(denoised,tr=tr,thsad=thsad,thsadc=thsadc,blksize=blksize,overlap=overlap,pel=pel,thscd1=thscd1,thscd2=thscd2,chromamv=chromamv,limit=limit,limitc=limitc)
        if show=='mde':
            return denoised

        if contrasharp>=2 or (contrasharp==1 and bm3dtyp=='no'):
            denoised=haf.ContraSharpening(denoised,sdr)
        if show=='mdecs':
            return denoised
    
    if not bm3dtyp=='no':
        if bm3dtyp=='cpu':
            BM3D=core.bm3dcpu.BM3D
        elif bm3dtyp=='cuda':
            BM3D=core.bm3dcuda.BM3D
        elif bm3dtyp=='cuda_rtc':
            BM3D=core.bm3dcuda_rtc.BM3D

        bdenoised=BM3D(denoised.fmtc.bitdepth(bits=32),sigma=sigma,radius=radius,block_step=block_step)
        if not radius==0:
            bdenoised=core.bm3d.VAggregate(bdenoised,radius,1)

        if finalest:
            sigma2=sigma2 if sigma2 else sigma
            radius2=radius2 if radius2 else radius
            bdenoised=BM3D(denoised.fmtc.bitdepth(bits=32),ref=bdenoised,sigma=sigma2,radius=radius2,block_step=max(block_step-1,1))
            if not radius2==0:
                bdenoised=core.bm3d.VAggregate(bdenoised,radius2,1)

        denoised=bdenoised.fmtc.bitdepth(bits=16)
        if show=='bm3d':
            return denoised

    if contrasharp>=1:
        denoised=haf.ContraSharpening(denoised,sdr)
    if show=='bm3dcs':
        return denoised

    if to709:
        denoised,sdr=[core.resize.Bicubic(i,transfer_in=1,transfer=16,nominal_luminance=nl) for i in (denoised,sdr)]
    
    output=core.std.Expr([src,sdr,denoised],'x y - z +') if to709 else denoised
    if lumaonly:
        output=core.std.ShufflePlanes([output,chromaclip],[0,1,2],vs.YUV)

    return output

#a simple mdegrain wrapper function that enough for my own use
def zmde(src,tr=2,thsad=100,thsadc=None,blksize=16,overlap=None,pel=1,chromamv=True,sharp=2,rfilter=4,truemotion=False,thscd1=400,thscd2=130,pref=None,**args):
    if thsadc==None:
        thsadc=thsad
    last=src
    if pref!=None:
        pass
    elif not chromamv:
        pref=core.resize.Bicubic(last,range_in_s='limited',range_s='full')
    else:
        pref=last
    if overlap==None:
        overlap=blksize//2
    
    sup=core.mv.Super(pref,hpad=blksize,vpad=blksize,sharp=sharp,rfilter=rfilter,pel=pel)
    sup2=core.mv.Super(last,hpad=blksize,vpad=blksize,sharp=sharp,levels=1,pel=pel)

    mvfw=core.mv.Analyse(sup,isb=False,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
    mvbw=core.mv.Analyse(sup,isb=True,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
    if tr>=2:
        mvfw2=core.mv.Analyse(sup,isb=False,delta=2,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
        mvbw2=core.mv.Analyse(sup,isb=True,delta=2,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
    if tr>=3:
        mvfw3=core.mv.Analyse(sup,isb=False,delta=3,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
        mvbw3=core.mv.Analyse(sup,isb=True,delta=3,blksize=blksize,overlap=overlap,truemotion=truemotion,chroma=chromamv)
    
    if tr==1:
        last=core.mv.Degrain1(last,sup2,mvbw,mvfw,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2,**args)
    elif tr==2:
        last=core.mv.Degrain2(last,sup2,mvbw,mvfw,mvbw2,mvfw2,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2,**args)
    elif tr>=3:
        last=core.mv.Degrain3(last,sup2,mvbw,mvfw,mvbw2,mvfw2,mvbw3,mvfw3,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2,**args)
    
    return last

#multi-pass f3kdb with optional contra-sharpening, masking and limit filter
#idea stolen from xyx98
def xdbcas(src,r=[8,15],y=[32,24],cb=[16,10],cr=[16,10],gy=[0,0],gc=[0,0],sm=[2,2],rs=[0,0],bf=[True,True],dg=[False,False],opt=[-1,-1],mt=[True,True],da=[3,3],ktv=[False,False],od=[16,16],rar=[1,1],rag=[1,1],rpr=[1,1],rpg=[1,1],passes=2,neo=True,casstr=0.3,mask=True,limit=True):
    last=db=src.fmtc.bitdepth(bits=16)
    r,y,cb,cr,gy,gc,sm,rs,bf,dg,opt,mt,da,ktv,od,rar,rag,rpr,rpg=[[i]*999 if isinstance(i,int) else i+[i[-1]]*999 for i in (r,y,cb,cr,gy,gc,sm,rs,bf,dg,opt,mt,da,ktv,od,rar,rag,rpr,rpg)]
    
    # l1,l2,l3,l4,l5,l6=[len(i) for i in (r,y,cb,cr,gy,gc)]
    # if l1==l2==l3==l4==l5==l6:
    #     passes=l6
    # else:
    #     passes=min(l1,l2,l3,l4,l5,l6)

    for i in range(passes):
        if neo:
            db=core.neo_f3kdb.Deband(db,r[i],y[i],cb[i],cr[i],gy[i],gc[i],sm[i],rs[i],bf[i],dg[i],opt[i],mt[i],da[i],ktv[i],od[i],rar[i],rag[i],rpr[i],rpg[i])
        else:
            db=core.f3kdb.Deband(db,r[i],y[i],cb[i],cr[i],gy[i],gc[i],sm[i],rs[i],bf[i],dg[i],opt[i],da[i],ktv[i],od[i],rar[i],rag[i],rpr[i],rpg[i])

    if isinstance(limit,bool) and limit:
        db=mvf.LimitFilter(db,last,thr=0.1,thrc=0.05,elast=20,planes=[0,1,2])
    elif callable(limit):
        db=limit(db,last)
    else:
        pass
    
    if mask:
        dbmask=xvs.mwdbmask(last)
        db=core.std.MaskedMerge(db,last,dbmask)

    if casstr==0:
        return db

    cas=core.cas.CAS(last,casstr,planes=[0,1,2])
    cas=mvf.LimitFilter(cas,last,thr=0.3,thrc=0.15,brighten_thr=0.15,elast=4,planes=[0,1,2])
    last=core.std.Expr([cas,last,db],'x y - z +')
    
    return last

#arbitrary crop, result resolution must be compatible with src clip subsampling tho
def arop(src,left=0,right=0,top=0,bottom=0): #mostly useless experimental function
    subsw,subsh=[2**src.format.subsampling_w,2**src.format.subsampling_h]
    if not (top%subsh or bottom%subsh or left%subsw or right%subsw):
        return core.std.Crop(src,left=left,right=right,top=top,bottom=bottom)
    else:
        l,r=[i if not i%subsw else 0 for i in (left,right)]
        t,b=[i if not i%subsw else 0 for i in (top,bottom)]

        last=core.std.Crop(src,left=l,right=r,top=t,bottom=b) if not(l==r==t==b==0) else src
        y,u,v=xvs.extractPlanes(last)

        l,r=[i if i%subsw else 0 for i in (left,right)]
        t,b=[i if i%subsw else 0 for i in (top,bottom)]

        y=core.std.Crop(y,left=l,right=r,top=t,bottom=b)
        w=(src.width-left-right)//subsw
        h=(src.height-top-bottom)//subsh
        src_left=l/subsw
        src_top=t/subsh
        src_width=(last.width-l-r)/subsw
        src_height=(last.height-t-b)/subsh
        u,v=[core.resize.Bicubic(i,w,h,src_left=src_left,src_top=src_top,src_width=src_width,src_height=src_height) for i in (u,v)]

        last=core.std.ShufflePlanes([y,u,v],[0,0,0],vs.YUV)
        return last

#padded finesharp, because finesharp like to mess up frames' edges
def pfinesharp(src,crop=True,psize=4,**args):
    sharpen=lambda x: finesharp.sharpen(x,**args)
    return rpfilter(src,filter=sharpen,psize=psize,crop=crop)

#nnrs and ssim is for chroma upscaling and downscaling only
def w2xaa(src,model=0,noise=-1,fp32=False,tile_size=0,format=None,full=None,matrix='709',nnrs=False,ssim=False,ssim_smooth=False,ssim_sigmoid=True,nnrs_down=None,ort=False,model_f=None,model_p=None,overlap=None):
    if full==None:
        try:
            full=not src.get_frame(0).props._ColorRange
        except:
            full=False
    src_range_s='full' if full else 'limited'
    src_format=src.format if format==None else format
    width,height=src.width,src.height
    precision=32 if fp32 else 16
    nnrs_down=nnrs if nnrs_down==None else nnrs_down
    if nnrs:
        last=Nnrs.nnedi3_resample(src,csp=vs.RGBS,fulls=full,mats=matrix)
    else:
        last=core.resize.Bicubic(src,format=vs.RGBS,range_in_s=src_range_s,matrix_in_s=matrix)

    if not ort:
        if tile_size==0:
            tile_size=[src.width,src.height]
        elif isinstance(tile_size,int):
            tile_size=[tile_size]*2
        last=core.w2xncnnvk.Waifu2x(last,model=model,scale=2,noise=noise,precision=precision,tile_w=tile_size[0],tile_h=tile_size[1])
    else:
        tile_size=[src.width,src.height] if tile_size==0 else tile_size
        overlap=4 if model==2 else 8 if overlap==None else overlap
        builtin=True if model_f==model_p==None else False
        model_f='waifu2x' if model_f==None else model_f
        model_g=['upconv_7_anime_style_art_rgb','upconv_7_photo','cunet'][model] if isinstance(model,int) else model
        model_n=f'noise{noise}_scale2.0x_model.onnx' if not noise==-1 else 'scale2.0x_model.onnx'
        model_p=os.path.join(model_f,model_g,model_n) if model_p==None else model_p
        last=core.ort.Model(last,model_p,provider='CUDA',builtin=builtin,fp16=not fp32,tilesize=tile_size,overlap=overlap)

    if ssim:
        last=muf.SSIM_downsample(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix,smooth=ssim_smooth,sigmoid=ssim_sigmoid)
    elif nnrs_down:
        last=Nnrs.nnedi3_resample(last,width,height,csp=src_format,fulld=full,matd=matrix)
    else:
        last=core.resize.Bicubic(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix)
    return last

#a workaround for amd rdna graphic cards to use knlmeanscl
def knl4a(src,planes=[1,1,1],rclip=None,h=1.2,amd=True,**args):
    if isinstance(h,list):
        if len(h)>=3:
            pass
        elif len(h)==2:
            h=[h[0],h[1],h[1]]
        else:
            h=[h[0],h[0],h[0]]
    else:
        h=[h,h,h]

    if not amd and planes==[1,1,1] and h[1]==h[2]:
        return src.knlm.KNLMeansCL(rclip=rclip,h=h[0],channels='Y',**args).knlm.KNLMeansCL(rclip=rclip,h=h[1],channels='UV',**args)

    y,u,v=xvs.extractPlanes(src)
    if amd:
        y,u,v=[core.std.ShufflePlanes([i,i,i],[0,0,0],vs.RGB) for i in (y,u,v)]
    if isinstance(rclip,vs.VideoNode):
        ry,ru,rv=[i(rclip) for i in (xvs.getY,xvs.getU,xvs.getV)]
        if amd:
            ry,ru,rv=[core.std.ShufflePlanes([i,i,i],[0,0,0],vs.RGB) for i in (ry,ru,rv)]
    else:
        ry=ru=rv=None
    if planes[0]:
        y=core.knlm.KNLMeansCL(y,rclip=ry,h=h[0],**args)
    if planes[1] and planes[2] and h[1]==h[2]:
        uv=core.knlm.KNLMeansCL(src,rclip=rclip,h=h[1],channels='UV',**args)
        u=xvs.getU(uv)
        v=xvs.getV(uv)
    else:
        if planes[1]:
            u=core.knlm.KNLMeansCL(u,rclip=ru,h=h[1],**args)
        if planes[2]:
            v=core.knlm.KNLMeansCL(v,rclip=rv,h=h[2],**args)
    y,u,v=[core.std.ShufflePlanes(i,[0],vs.GRAY) for i in (y,u,v)]
    return core.std.ShufflePlanes([y,u,v],[0,0,0],vs.YUV)

#line mask?
def wtfmask(src,nnrs=True,t_l=16,t_h=26,range='limited',op=[1],optc=1,bthr=1,**args):
    if nnrs:
        last=Nnrs.nnedi3_resample(src,csp=vs.RGBS)
    else:
        last=core.resize.Bicubic(src,format=vs.RGBS,matrix_in=1)
    last=core.tcanny.TCanny(last,t_l=t_l,t_h=t_h,op=optc,**args)
    if range in ['full','pc']:
        last=last.resize.Bicubic(format=vs.GRAY16,matrix=1,range_s='full').std.Binarize(256*bthr,0,65535)
    else:
        last=last.resize.Bicubic(format=vs.GRAY16,matrix=1,range_s='limited').std.Binarize(256*(16+(235-16)/256*bthr),16*256,235*256)
    f=[core.std.Minimum,core.std.Maximum,core.std.Deflate,core.std.Inflate]
    for i in op:
        last=f[i](last)
    return last


def bordermask(src,l=0,r=0,t=0,b=0,d=16):
    return core.std.BlankClip(src,format=core.std.BlankClip(format=vs.GRAY16).fmtc.bitdepth(bits=d).format,color=0).std.Crop(left=l,right=r,top=t,bottom=b).std.AddBorders(left=l,right=r,top=t,bottom=b,color=2**d-1 if d<32 else 1)

#resize padded filter
def rpfilter(input,ref=None,other=None,filter=lambda x: x,psize=2,crop=True):
    input=rpclip(input,psize)
    if isinstance(other,dict):
        for key in other:
            if isinstance(other[key],vs.VideoNode):
                other[key]=rpclip(other[key],psize)
    if isinstance(ref,vs.VideoNode):
        ref=rpclip(ref,psize)
        if isinstance(other,dict):
            last=filter(input,ref,**other)
        else:
            last=filter(input,ref)
    else:
        if isinstance(other,dict):
            last=filter(input,**other)
        else:
            last=filter(input)
    if crop:
        last=core.std.Crop(last,*[psize]*4)
    return last

#resize pad clip
def rpclip(input,psize=2):
    return core.resize.Bicubic(input,input.width+2*psize,input.height+2*psize,src_top=-psize,src_left=-psize,src_width=input.width+2*psize,src_height=input.height+2*psize)

#nnedi3 preview
def n3pv(*args,**kwargs):
    scale=kwargs.get('scale') if kwargs.get('scale')!=None else 2
    nns=kwargs.get('nns') if kwargs.get('nns')!=None else 1
    nsize=kwargs.get('nsize') if kwargs.get('nsize')!=None else 0
    qual=kwargs.get('qual') if kwargs.get('qual')!=None else 1
    mode=kwargs.get('mode') if kwargs.get('mode')!=None else nnrs_mode_default
    last=list()
    if len(args)==1:
        if isinstance(args[0],list):
            for clip in args[0]:
                last.append(Nnrs.nnedi3_resample(clip,clip.width*scale,clip.height*scale,csp=vs.RGB24,nns=nns,nsize=nsize,qual=qual,mode=mode))
        elif isinstance(args[0],vs.VideoNode):
            last.append(Nnrs.nnedi3_resample(args[0],args[0].width*scale,args[0].height*scale,csp=vs.RGB24,nns=nns,nsize=nsize,qual=qual,mode=mode))
        else:
            raise TypeError('input for preview should be list or clip')
    else:
        for i in range(len(args)):
            last.append(Nnrs.nnedi3_resample(args[i],args[i].width*scale,args[i].height*scale,csp=vs.RGB24,nns=nns,nsize=nsize,qual=qual,mode=mode).sub.Subtitle('clip%d'%i))
    return core.std.Interleave(last)

#quack quack, I'll take your grains
#a dumb-ass func may be suitable for old movies with heavy dynamic grains
def quack(src,knl={},md1={},bm1={},md2={},bm2={}):
    _knl={'amd':False,'a':1,'s':2,'d':3,'h':2}
    _md1={'thsad':250,'thscd1':250,'limit':768,'tr':3}
    _bm1={'sigma':[2,2,2],'radius':1}
    _md2={'thsad':250,'thscd1':250,'limit':768,'tr':3}
    _bm2={'sigma':[1,1,1],'radius':1}
    _knl.update(knl)
    _md1.update(md1)
    _bm1.update(bm1)
    _md2.update(md2)
    _bm2.update(bm2)
    if src.format.bits_per_sample!=16:
        src=src.fmtc.bitdepth(bits=16)
    last=src
    m=last.std.Median()
    n=knl4a(last,rclip=m,**_knl)
    last=zmde(last,pref=n,**_md1)
    last=bm3d(last,iref=src,**_bm1)
    last=zmde(src,pref=last,**_md2)
    last=bm3d(last,iref=src,**_bm2)
    return last

#use y channel or opponent chroma channel as reference to repair uv channels with bilateral
#tbilateral is much trickier to use, the "ref" doesn't even mean the same thing, just add it for testing
#parameters you should really care about are ones in first line
def bilateraluv(src,ch='uv',mode='down',method='spline36',S=1,R=0.02,lumaref=True,crossref=False,\
    algo=0,P=None,T=False,diameter=3,sdev=0.5,idev=0.01,cs=1,d2=True,kerns=1,kerni=1,restype=0,**args):
    if mode.lower()=='up':
        targetw=src.width
        targeth=src.height
    elif mode.lower()=='down':
        targetw=src.width / (2**src.format.subsampling_w)
        targeth=src.height / (2**src.format.subsampling_h)
    else:
        raise ValueError('mode not supported')

    if lumaref:
        if method.lower()=='nnrs':
            last=Nnrs.nnedi3_resample(src,targetw,targeth,csp=vs.YUV444P16)
        elif method.lower() in ['point','bilinear','bicubic','lanczos','spline16','spline36','spline64']:
            resizers={'point':core.resize.Point,'bilinear':core.resize.Bilinear,'bicubic':core.resize.Bicubic,'lanczos':core.resize.Lanczos,'spline16':core.resize.Spline16,'spline36':core.resize.Spline36,'spline64':core.resize.Spline64}
            resizer=resizers[method.lower()]
            last=resizer(src,targetw,targeth,format=vs.YUV444P16,**args)
        else:
            raise ValueError('resize method not supported')
    else:
        last=src

    y,u,v=xvs.extractPlanes(last)
    if 'u' in ch.lower():
        if lumaref:
            ub=core.bilateral.Bilateral(u,y,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(u,y,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
        else:
            ub=u
        if crossref:
            ub=core.bilateral.Bilateral(ub,v,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(ub,v,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
    if 'v' in ch.lower():
        if lumaref:
            vb=core.bilateral.Bilateral(v,y,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(v,y,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
        else:
            vb=v
        if crossref:
            vb=core.bilateral.Bilateral(vb,u,sigmaS=S,sigmaR=R,algorithm=algo,PBFICnum=P) if not T else\
                core.tbilateral.TBilateral(vb,u,diameter,sdev,idev,cs,d2,kerns,kerni,restype)
    return core.std.ShufflePlanes([src,ub,vb],[0,0,0],vs.YUV)

########################################################
########## HERE STARTS THE COPY-PASTE SECTION ##########
########################################################

#copy-paste from xyx98's xvs with some modification
#new feature
#radius2: similar to "sigma2"
#iterates: outputs all bm3d passes as a list in the order they take place
#iref: initial ref
#keepfloat: like what it said
#vt: v-bm3d type, 0 for good old bm3dcuda+bm3d.VAggregate, 1 for bm3dcuda.BM3Dv2
def bm3d(clip:vs.VideoNode,iref=None,sigma=[3,3,3],sigma2=None,preset="fast",preset2=None,mode=bm3d_mode_default,radius=0,radius2=None,chroma=False,fast=True,
            block_step1=None,bm_range1=None, ps_num1=None, ps_range1=None,
            block_step2=None,bm_range2=None, ps_num2=None, ps_range2=None,
            extractor_exp=0,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT",
            refine=1,dmode=0,iterates=False,keepfloat=False,vt=0):
    bits=clip.format.bits_per_sample
    clip=core.fmtc.bitdepth(clip,bits=32)
    iref=core.fmtc.bitdepth(iref,bits=32) if isinstance(iref,vs.VideoNode) else None
    if chroma is True and clip.format.id !=vs.YUV444PS:
        raise ValueError("chroma=True only works on yuv444")
    
    if radius2 is None:
        radius2=radius
    isvbm3d=radius+radius2>0

    if sigma2 is None:
        sigma2=sigma

    if preset2 is None:
        preset2=preset

    if preset not in ["fast","lc","np","high"] or preset2 not in ["fast","lc","np","high"]:
        raise ValueError("preset and preset2 must be 'fast','lc','np',or'high'")

    parmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,9,2,4],
        "lc"  :[6,9,2,4],
        "np"  :[4,16,2,5],
        "high":[3,16,2,7],
    }

    vparmas1={
        #block_step,bm_range, ps_num, ps_range
        "fast":[8,7,2,4],
        "lc"  :[6,9,2,4],
        "np"  :[4,12,2,5],
        "high":[3,16,2,7],
    }

    parmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,9,2,5],
        "lc"  :[5,9,2,5],
        "np"  :[3,16,2,6],
        "high":[2,16,2,8],
    }

    vparmas2={
        #block_step,bm_range, ps_num, ps_range
        "fast":[7,7,2,5],
        "lc"  :[5,9,2,5],
        "np"  :[3,12,2,6],
        "high":[2,16,2,8],
    }


    if isvbm3d:
        p1,p2=vparmas1,vparmas2
    else:
        p1,p2=parmas1,parmas2

    
    block_step1=p1[preset][0] if block_step1 is None else block_step1
    bm_range1=p1[preset][1] if bm_range1 is None else bm_range1
    ps_num1=p1[preset][2] if ps_num1 is None else ps_num1
    ps_range1=p1[preset][3] if ps_range1 is None else ps_range1

    block_step2=p2[preset2][0] if block_step2 is None else block_step2
    bm_range2=p2[preset2][1] if bm_range2 is None else bm_range2
    ps_num2=p2[preset2][2] if ps_num2 is None else ps_num2
    ps_range2=p2[preset2][3] if ps_range2 is None else ps_range2

    if iterates:    
        outputs=list()
    if isvbm3d:
        flt=bm3d_core(clip,ref=iref,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
        if radius>0 and vt==0:
            flt=core.bm3d.VAggregate(flt,radius=radius,sample=1)
        if iterates:
            outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

        for i in range(refine):
            flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius2,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
            if radius2>0 and vt==0:
                flt=core.bm3d.VAggregate(flt,radius=radius2,sample=1)
            if iterates:
                outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

    else:
        flt=bm3d_core(clip,ref=iref,mode=mode,sigma=sigma,radius=radius,block_step=block_step1,bm_range=bm_range1,ps_num=ps_num1,ps_range=ps_range1,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
        if iterates:
            outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

        for i in range(refine):
            flt=bm3d_core(clip,ref=flt,mode=mode,sigma=sigma2,radius=radius2,block_step=block_step2,bm_range=bm_range2,ps_num=ps_num2,ps_range=ps_range2,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s,vt=vt)
            if iterates:
                outputs.append(core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt)

    return core.fmtc.bitdepth(flt,bits=bits,dmode=dmode) if not keepfloat else flt if not iterates else outputs

#copy-paste from xyx98's xvs
def bm3d_core(clip,ref=None,mode="cpu",sigma=3.0,block_step=8,bm_range=9,radius=0,ps_num=2,ps_range=4,chroma=False,fast=True,extractor_exp=0,device_id=0,bm_error_s="SSD",transform_2d_s="DCT",transform_1d_s="DCT",vt=0):
    if mode not in ["cpu","cuda","cuda_rtc"]:
        raise ValueError("mode must be cpu,or cuda,or cuda_rtc")
    elif mode=="cpu":
        if vt==1:
            return core.bm3dcpu.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
        return core.bm3dcpu.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma)
    elif mode=="cuda":
        if vt==1:
            return core.bm3dcuda.BM3Dv2(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
        return core.bm3dcuda.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id)
    else:
        if vt==1:
            return core.bm3dcuda_rtc.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)
        return core.bm3dcuda_rtc.BM3D(clip,ref=ref,sigma=sigma,block_step=block_step,bm_range=bm_range,radius=radius,ps_num=ps_num,ps_range=ps_range,chroma=chroma,fast=fast,extractor_exp=extractor_exp,device_id=device_id,bm_error_s=bm_error_s,transform_2d_s=transform_2d_s,transform_1d_s=transform_1d_s)

#copy-paste from xyx98's xvs with some modification
#new feature
#mask_gen_clip: an alternative clip can be provided for diff mask generation
#mask_operate_func: a function can be specified for mask operations after generation (e.g. expand, inpand and more)
#linear, sigmoid: do descale in linear or sigmoid light
def rescale(src:vs.VideoNode,kernel:str,w=None,h=None,mask=True,mask_dif_pix=2,show="result",postfilter_descaled=None,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,**args):
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")
    
    src=core.fmtc.bitdepth(src,bits=16)
    luma=xvs.getY(src)
    tin='709' if args.get("tin") is None else args.get("tin")
    min='709' if args.get("min") is None else args.get("min")
    if linear or sigmoid:
        luma=core.resize.Bicubic(luma,transfer_in_s=tin,transfer_s='linear',matrix_in_s=min)
        if sigmoid:
            luma=haf.SigmoidInverse(luma)
    if isinstance(mask_gen_clip,vs.VideoNode):
        luma=core.std.Interleave([luma,xvs.getY(mask_gen_clip)])
    ####
    if kernel in ["Debilinear","Despline16","Despline36","Despline64"]:
        luma_de=eval("core.descale.{k}(luma.fmtc.bitdepth(bits=32),w,h)".format(k=kernel))
        luma_up=eval("core.resize.{k}(luma_de,src_w,src_h)".format(k=kernel[2:].capitalize())).fmtc.bitdepth(bits=16,dmode=1)
    elif kernel=="Debicubic":
        luma_de=core.descale.Debicubic(luma.fmtc.bitdepth(bits=32),w,h,b=args.get("b"),c=args.get("c"))
        luma_up=core.resize.Bicubic(luma_de,src_w,src_h,filter_param_a=args.get("b"),filter_param_b=args.get("c")).fmtc.bitdepth(bits=16,dmode=1)
    else:
        luma_de=core.descale.Delanczos(luma.fmtc.bitdepth(bits=32),w,h,taps=args.get("taps"))
        luma_up=core.resize.Lanczos(luma_de,src_w,src_h,filter_param_a=args.get("taps")).fmtc.bitdepth(bits=16,dmode=1)
    
    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=luma_up[1::2]
        luma=luma[::2]
        luma_de=luma_de[::2]
        luma_up=luma_up[::2]

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        luma_de=postfilter_descaled(luma_de)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")
    mode=nnrs_mode_default if args.get("mode") is None else args.get("mode")

    if linear or sigmoid:
        if sigmoid:
            luma_de=haf.SigmoidDirect(luma_de.fmtc.bitdepth(bits=16))
        luma_de=core.resize.Bicubic(luma_de,transfer_in_s='linear',transfer_s=tin,matrix_s=min)
    luma_rescale=nnrs.nnedi3_resample(luma_de,src_w,src_h,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,mode=mode).fmtc.bitdepth(bits=16)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip,mclip_up=luma,luma_up
        mask=core.std.Expr([mclip,mclip_up],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])

        luma_rescale=core.std.MaskedMerge(luma_rescale,xvs.getY(src),mask)
    
    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def rescalef(src: vs.VideoNode,kernel: str,w=None,h=None,bh=None,bw=None,mask=True,mask_dif_pix=2,show="result",postfilter_descaled=None,selective=False,upper=0.0001,lower=0.00001,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,linear=False,sigmoid=False,**args):
    #for decimal resolution descale,refer to GetFnative
    if src.format.color_family not in [vs.YUV,vs.GRAY]:
        raise ValueError("input clip should be YUV or GRAY!")

    src_h,src_w=src.height,src.width
    if w is None and h is None:
        w,h=1280,720
    elif w is None:
        w=int(h*src_w/src_h)
    else:
        h=int(w*src_h/src_w)

    if bh is None:
        bh=1080

    if w>=src_w or h>=src_h:
        raise ValueError("w,h should less than input resolution")
    
    kernel=kernel.strip().capitalize()
    if kernel not in ["Debilinear","Debicubic","Delanczos","Despline16","Despline36","Despline64"]:
        raise ValueError("unsupport kernel")

    src=core.fmtc.bitdepth(src,bits=16)
    luma=xvs.getY(src)
    tin='709' if args.get("tin") is None else args.get("tin")
    min='709' if args.get("min") is None else args.get("min")
    if linear or sigmoid:
        luma=core.resize.Bicubic(luma,transfer_in_s=tin,transfer_s='linear',matrix_in_s=min)
        if sigmoid:
            luma=haf.SigmoidInverse(luma)
    if isinstance(mask_gen_clip,vs.VideoNode):
        luma=core.std.Interleave([luma,xvs.getY(mask_gen_clip)])
    cargs=xvs.cropping_args(src.width,src.height,h,bh,bw)
    ####
    if kernel in ["Debilinear","Despline16","Despline36","Despline64"]:
        luma_de=eval("core.descale.{k}(luma.fmtc.bitdepth(bits=32),**cargs.descale_gen())".format(k=kernel))
        luma_up=eval("core.resize.{k}(luma_de,**cargs.resize_gen())".format(k=kernel[2:].capitalize()))
    elif kernel=="Debicubic":
        luma_de=core.descale.Debicubic(luma.fmtc.bitdepth(bits=32),b=args.get("b"),c=args.get("c"),**cargs.descale_gen())
        luma_up=core.resize.Bicubic(luma_de,filter_param_a=args.get("b"),filter_param_b=args.get("c"),**cargs.resize_gen())
    else:
        luma_de=core.descale.Delanczos(luma.fmtc.bitdepth(bits=32),taps=args.get("taps"),**cargs.descale_gen())
        luma_up=core.resize.Lanczos(luma_de,filter_param_a=args.get("taps"),**cargs.resize_gen())#

    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=luma_up[1::2]
        luma=luma[::2]
        luma_de=luma_de[::2]
        luma_up=luma_up[::2]

    diff = core.std.Expr([luma.fmtc.bitdepth(bits=32), luma_up], f'x y - abs dup 0.015 > swap 0 ?').std.Crop(10, 10, 10, 10).std.PlaneStats()

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        luma_de=postfilter_descaled(luma_de)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")#keep behavior before
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")#keep behavior before
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")

    if linear or sigmoid:
        if sigmoid:
            luma_de=haf.SigmoidDirect(luma_de.fmtc.bitdepth(bits=16))
        luma_de=core.resize.Bicubic(luma_de,transfer_in_s='linear',transfer_s=tin,matrix_s=min)
    luma_rescale=nnrs.nnedi3_resample(luma_de,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp,**cargs.nnrs_gen()).fmtc.bitdepth(bits=16)

    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]
        return fout

    luma_rescale=core.std.ModifyFrame(luma_rescale,[diff,luma_rescale],calc)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip,mclip_up=luma,luma_up
        mask=core.std.Expr([mclip,mclip_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])

        luma_rescale=core.std.MaskedMerge(luma_rescale,xvs.getY(src),mask)
    
    if selective:
        base=upper-lower
        #x:rescale y:src
        expr=f"x.diff {upper} > y x.diff {lower} < x {upper} x.diff -  {base} / y * x.diff {lower} - {base} / x * + ? ?"
        luma_rescale=core.akarin.Expr([luma_rescale,luma], expr)

    if show=="descale":
        return luma_de
    elif show=="mask":
        return mask
    elif show=="both":
        return luma_de,mask
    elif show=="diff":
        return core.text.FrameProps(luma_rescale,"diff", scale=2)

    if src.format.color_family==vs.GRAY:
        return luma_rescale
    else:
        return core.std.ShufflePlanes([luma_rescale,src],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def multirescale(clip:vs.VideoNode,kernels:list[dict],w:Optional[int]=None,h:Optional[int]=None,mask:bool=True,mask_dif_pix:float=2.5,postfilter_descaled=None,selective_disable:bool=False,disable_thr:float=0.00001,showinfo=False,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,**args):
    clip=core.fmtc.bitdepth(clip,bits=16)
    luma=xvs.getY(clip)
    src_h,src_w=clip.height,clip.width
    def getwh(w,h):
        if w is None and h is None:
            w,h=1280,720
        elif w is None:
            w=int(h*src_w/src_h)
        elif h is None:
            h=int(w*src_h/src_w)

        if w>=src_w or h>=src_h:
            raise ValueError("w,h should less than input resolution")
        return w,h

    w,h=getwh(w,h)

    info_gobal=f"gobal:\nresolution:{w}x{h}\tmask:{mask}\tmask_dif_pix:{mask_dif_pix}\tpostfilter_descaled:{'yes' if postfilter_descaled else 'no'}\nselective_disable:{selective_disable}\tdisable_thr:{disable_thr:f}\nextra:{str(args)}"
    rescales=[]
    total=len(kernels)
    for i in kernels:
        k=i["k"][2:]
        kb,kc,ktaps=i.get("b"),i.get("c"),i.get("taps")
        kw,kh=i.get("w"),i.get("h")
        if kw is not None or kh is not None:
            kw,kh=getwh(kw,kh)
        else:
            kw,kh=w,h
        kmask=mask if i.get("mask") is None else i.get("mask")
        kmdp=mask_dif_pix if i.get("mask_dif_pix") is None else i.get("mask_dif_pix")
        kpp=postfilter_descaled if i.get("postfilter_descaled") is None else i.get("postfilter_descaled")
        multiple=1 if i.get("multiple") is None else i.get("multiple")
        mthr=mthr if i.get("mthr") is None else i.get("mthr")
        mgc=mask_gen_clip if i.get("mask_gen_clip") is None else i.get("mask_gen_clip")
        mof=mask_operate_func if i.get("mask_operate_func") is None else i.get("mask_operate_func")

        rescales.append(MRcore(luma,kernel=k,w=kw,h=kh,mask=kmask,mask_dif_pix=kmdp,postfilter_descaled=kpp,taps=ktaps,b=kb,c=kc,multiple=multiple,mthr=mthr,mask_gen_clip=mgc,mask_operate_func=mof,**args))


    def selector(n,f,src,clips):
        kernels_info=[]
        index,mindiff=0,f[0].props["diff"]
        for i in range(total):
            tmpdiff=f[i].props["diff"]
            kernels_info.append(f"kernel {i}:\t{kernels[i]}\n{tmpdiff:.10f}")
            if tmpdiff<mindiff:
                index,tmpdiff=i,tmpdiff

        info=info_gobal+"\n--------------------\n"+("\n--------------------\n").join(kernels_info)+"\n--------------------\ncurrent usage:\n"
        if selective_disable and mindiff>disable_thr:
            last=src
            info+="source"
        else:
            last=clips[index]
            info+=kernels_info[index]
        if showinfo:
            last=core.text.Text(last,info.replace("\t","    "))
        return last.std.SetFrameProp('kindex',intval=index)

    last=core.std.FrameEval(luma,partial(selector,src=luma,clips=rescales),prop_src=rescales)
    if clip.format.color_family==vs.GRAY:
        return last
    else:
        return core.std.ShufflePlanes([last,clip],[0,1,2],vs.YUV)

#copy-paste from xyx98's xvs with some modification
def MRcore(clip:vs.VideoNode,kernel:str,w:int,h:int,mask: bool=True,mask_dif_pix:float=2,postfilter_descaled=None,taps:int=3,b:float=0,c:float=0.5,multiple:float=1,mthr:list[int]=[2,2],mask_gen_clip=None,mask_operate_func=None,**args):
    src_w,src_h=clip.width,clip.height
    clip32=core.fmtc.bitdepth(clip,bits=32)
    if isinstance(mask_gen_clip,vs.VideoNode):
        clip32=core.std.Interleave([clip32,xvs.getY(mask_gen_clip).fmtc.bitdepth(bits=32)])
    descaled=core.descale.Descale(clip32,width=w,height=h,kernel=kernel.lower(),taps=taps,b=b,c=c)
    upscaled=xvs.resize_core(kernel.capitalize(),taps,b,c)(descaled,src_w,src_h)
    if isinstance(mask_gen_clip,vs.VideoNode):
        mclip=xvs.getY(mask_gen_clip)
        mclip_up=upscaled[1::2]
        clip32=clip32[::2]
        descaled=descaled[::2]
        upscaled=upscaled[::2]
    diff=core.std.Expr([clip32,upscaled],"x y - abs dup 0.015 > swap 0 ?").std.PlaneStats()
    
    def calc(n,f): 
        fout=f[1].copy()
        fout.props["diff"]=f[0].props["PlaneStatsAverage"]*multiple
        return fout

    if postfilter_descaled is None:
        pass
    elif callable(postfilter_descaled):
        descaled=postfilter_descaled(descaled)
    else:
        raise ValueError("postfilter_descaled must be a function")

    nsize=3 if args.get("nsize") is None else args.get("nsize")
    nns=args.get("nns")
    qual=2 if args.get("qual") is None else args.get("qual")
    etype=args.get("etype")
    pscrn=1 if args.get("pscrn") is None else args.get("pscrn")
    exp=args.get("exp")

    rescale=nnrs.nnedi3_resample(descaled,src_w,src_h,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,exp=exp).fmtc.bitdepth(bits=16)

    if mask:
        if not isinstance(mask_gen_clip,vs.VideoNode):
            mclip=clip
            mclip_up=upscaled
        mask=core.std.Expr([mclip,mclip_up.fmtc.bitdepth(bits=16,dmode=1)],"x y - abs").std.Binarize(mask_dif_pix*256)
        if callable(mask_operate_func):
            mask=mask_operate_func(mask)
        else:
            mask=xvs.expand(mask,cycle=mthr[0])
            mask=xvs.inpand(mask,cycle=mthr[1])
        rescale=core.std.MaskedMerge(rescale,clip,mask)

    return core.std.ModifyFrame(rescale,[diff,rescale],calc)

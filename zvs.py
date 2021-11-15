import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import xvs
import mvsfunc as mvf
import finesharp
import muvsfunc as muf

def pqdenoise(src,sigma=[1,1,1],lumaonly=False,block_step=7,radius=1,finalest=False,bm3dtyp='cpu',mdegrain=True,tr=2,pel=1,blksize=16,overlap=None,chromamv=True,thsad=100,thsadc=None,thscd1=400,thscd2=130,nl=100,contrasharp=1,to709=1,show='output'):
    if lumaonly:
        chromamv=False
        chromaclip=src
        src=xvs.getY(src)

    src=src.fmtc.bitdepth(bits=16)
    denoised=sdr=core.resize.Bicubic(src,transfer_in=16,transfer=1,nominal_luminance=nl) if to709 else src
    if mdegrain:
        denoised=zmde(denoised,tr=tr,thsad=thsad,thsadc=thsadc,blksize=blksize,overlap=overlap,pel=pel,thscd1=thscd1,thscd2=thscd2,chromamv=chromamv)
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
            bdenoised=BM3D(denoised.fmtc.bitdepth(bits=32),ref=bdenoised,sigma=sigma,radius=radius,block_step=max(block_step-1,1))
            if not radius==0:
                bdenoised=core.bm3d.VAggregate(bdenoised,radius,1)

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


def zmde(src,tr=2,thsad=100,thsadc=None,blksize=16,overlap=None,pel=1,chromamv=True,sharp=2,rfilter=4,truemotion=False,thscd1=400,thscd2=130,pref=None):
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
        last=core.mv.Degrain1(last,sup2,mvbw,mvfw,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2)
    elif tr==2:
        last=core.mv.Degrain2(last,sup2,mvbw,mvfw,mvbw2,mvfw2,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2)
    elif tr>=3:
        last=core.mv.Degrain3(last,sup2,mvbw,mvfw,mvbw2,mvfw2,mvbw3,mvfw3,thsad=thsad,thsadc=thsadc,thscd1=thscd1,thscd2=thscd2)
    
    return last


def xdbcas(src,r=[8,15],y=[32,24],cb=[16,10],cr=[16,10],gy=[0,0],gc=[0,0],neo=False,casstr=0.7):
    last=db=src.fmtc.bitdepth(bits=16)
    r,y,cb,cr,gy,gc=[list(i) if isinstance(i,int) else i for i in (r,y,cb,cr,gy,gc)]
    if neo:
        f3k=core.neo_f3kdb.Deband
    else:
        f3k=core.f3kdb.Deband
    
    l1,l2,l3,l4,l5,l6=[len(i) for i in (r,y,cb,cr,gy,gc)]
    if l1==l2==l3==l4==l5==l6:
        passes=l6
    else:
        passes=min(l1,l2,l3,l4,l5,l6)

    for i in range(passes):
        db=f3k(db,r[i],y[i],cb[i],cr[i],gy[i],gc[i],output_depth=16)
    db=mvf.LimitFilter(db,last,thr=0.1,thrc=0.05,elast=20,planes=[0,1,2])
    dbmask=xvs.mwdbmask(last)
    db=core.std.MaskedMerge(db,last,dbmask)

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
        y,u,v=[i(last) for i in (xvs.getY,xvs.getU,xvs.getV)]

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
def pfinesharp(src,crop=False,**args):
    last=core.resize.Bicubic(src,src.width+8,src.height+8,src_top=-4,src_left=-4,src_width=src.width+8,src_height=src.height+8)
    last=finesharp.sharpen(last,**args)
    if crop:
        last=core.std.Crop(last,4,4,4,4)
    return last

#nnrs and ssim is for chroma upscaling and downscaling only
def w2xaa(src,model=0,noise=-1,fp32=False,tile_size=0,format=None,full=None,matrix='709',nnrs=False,ssim=False,ssim_smooth=False,ssim_sigmoid=True,nnrs_down=None):
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
        last=xvs.nnrs.nnedi3_resample(src,csp=vs.RGBS,fulls=full,mats=matrix)
    else:
        last=core.resize.Bicubic(src,format=vs.RGBS,range_in_s=src_range_s,matrix_in_s=matrix)
    last=core.w2xnvk.Waifu2x(last,model=model,scale=2,noise=noise,precision=precision,tile_size=tile_size)
    if ssim:
        last=muf.SSIM_downsample(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix,smooth=ssim_smooth,sigmoid=ssim_sigmoid)
    elif nnrs_down:
        last=xvs.nnrs.nnedi3_resample(last,width,height,csp=src_format,fulld=full,matd=matrix)
    else:
        last=core.resize.Bicubic(last,width,height,format=src_format,range_s=src_range_s,matrix_s=matrix)
    return last

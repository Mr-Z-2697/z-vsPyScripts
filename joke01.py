import vapoursynth as vs
core=vs.core
import zvs
__all__=['lrnoise','debit']

def lrnoise(src,lr=(1280,720),gy=50,gc=0,hc=0,vc=0,con=0,seed=1,opt=0,a1=20,adg=False,mdg=False,azmdg={},cdif=False,fnoise=None):
    src=src.fmtc.bitdepth(bits=16)
    za={'thsad':1000,'truemotion':True}
    za.update(azmdg)
    last=src
    lr=last.fmtc.resample(lr[0],lr[1])
    if callable(fnoise):
        lrn=fnoise(lr)
    else:
        lrn=core.grain.Add(lr,var=gy,uvar=gc,hcorr=hc,vcorr=vc,constant=con,seed=seed,opt=opt)
    if adg:
        lr=lr.std.PlaneStats()
        lrm=lr.adg.Mask(adg)
        lrn=core.std.MaskedMerge(lr,lrn,lrm)
    if mdg: mvd=zvs.zmdg(lr,mvout=True,**za)
    if cdif:
        nd=zvs.cdif(lrn,lr)
    else:
        nd=core.std.MakeDiff(lrn,lr)
    if mdg: nd=zvs.zmdg(nd,mvin=mvd,**za)
    ndhr=nd.fmtc.resample(last.width,last.height,kernel='gaussian',a1=a1)
    if cdif:
        last=zvs.cdif(last,ndhr,1)
    else:
        last=core.std.MergeDiff(last,ndhr)
    return last

# i know mvf.Depth can do just want a simpler approach
# is it still the depth we want if chroma sign is considered? what about the range of values?
# brain cells are dying
# cs: consider chroma sign
# cs2: is a free upgrade to CS:GO
# cs2: half the scaling for +-0.5 range
def debit(src,depth=1,dither=0,fulls=None,fulld=None,cs=False,cs2=False,count=None):
    if depth>=8 or (count!=None and count>=256):
        raise ValueError("use normal dither bro")
    isrgb=src.format.color_family==vs.RGB
    if fulls==None: fulls=True if isrgb else False
    if fulld==None: fulld=True if isrgb else False
    if src.format.sample_type==vs.FLOAT:
        src=core.resize.Point(src,format=src.format.replace(sample_type=vs.INTEGER,bits_per_sample=16),range_s='full')
        fulls=True
    if src.format.bits_per_sample < 16:
        src=zvs.simplebitdepth(src,16)
    if isinstance(dither,int):
        rehtid=lambda x:x.fmtc.bitdepth(bits=8,dmode=dither)
    elif isinstance(dither,str):
        rehtid=lambda x:x.resize.Point(format=x.format.replace(bits_per_sample=8),dither_type=dither)
    elif callable(dither):
        rehtid=dither
    if count==None:
        count=2**depth
    scaling=255/(count-1)
    scalingc=scaling if not cs2 else scaling/2
    last=src
    if not fulls:
        last=zvs.setrange(last,'rm')
        last=last.resize.Point(range_in_s='limited',range_s='full')
    if cs:
        last=last.std.Expr([f'x {scaling} /',f'x 32768 - {scalingc} / 32768 +'])
    else:
        last=last.std.Expr(f'x {scaling} /')
    last=rehtid(last)
    if cs:
        last=last.std.Expr([f'x {scaling} *',f'x 128 - {scalingc} * 128 +'])
    else:
        last=last.std.Expr(f'x {scaling} *')
    if not fulld:
        last=zvs.setrange(last,'rm')
        last=last.resize.Point(range_in_s='full',range_s='limited')
    return last
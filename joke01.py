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
def debit(src,depth=1,dither=0,fulls=False,fulld=False):
    if src.format.sample_type==vs.FLOAT:
        raise NotImplementedError("panik")
    if src.format.bits_per_sample < 16:
        src=core.std.Expr(src,'x {} *'.format(2**(16-src.format.bits_per_sample)),src.format.replace(bits_per_sample=16))
    if isinstance(dither,int):
        rehtid=lambda x:x.fmtc.bitdepth(bits=8,dmode=dither)
    elif isinstance(dither,str):
        rehtid=lambda x:x.resize.Point(format=x.format.replace(bits_per_sample=8),dither_type=dither)
    elif callable(dither):
        rehtid=dither
    scaling=255/(2**depth-1)
    last=src
    if not fulls:
        last=zvs.setrange(last,'rm')
        last=last.resize.Point(range_in_s='limited',range_s='full')
    last=last.std.Expr(f'x {scaling} /')
    last=rehtid(last)
    last=last.std.Expr(f'x {scaling} *')
    if not fulld:
        last=zvs.setrange(last,'rm')
        last=last.resize.Point(range_in_s='full',range_s='limited')
    return last
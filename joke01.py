import vapoursynth as vs
core=vs.core
import zvs
__all__=['lrnoise']

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
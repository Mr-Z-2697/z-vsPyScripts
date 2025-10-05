import vapoursynth as vs
core=vs.core
import zvs
__all__=['lrnoise','debit','naive32k','naive32i']

def lrnoise(src,lr=(1280,720),gy=50,gc=0,hc=0,vc=0,con=0,seed=1,opt=0,a1=20,adg=False,mdg=False,azmdg={},cdif=False,fnoise=None):
    src=zvs.simplebitdepth(src,16)
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
def debit(src,depth=1,dither=0,fulls=None,fulld=None,cs=None,cs2=None,count=None):
    if depth>=8 or (count!=None and count>=256):
        raise ValueError("use normal dither for >8bit bro")
    isrgb=src.format.color_family==vs.RGB
    if fulls==None: fulls=True if isrgb else False
    if fulld==None: fulld=True if isrgb else False
    if cs==None: cs=False if isrgb else True
    if cs2==None: cs2=False
    if src.format.sample_type==vs.FLOAT:
        src=core.resize.Point(src,format=src.format.replace(sample_type=vs.INTEGER,bits_per_sample=16),range_s='full')
        fulls=True
    if src.format.bits_per_sample < 16:
        src=zvs.simplebitdepth(src,16)
    if isinstance(dither,int):
        rehtid=lambda x:x.fmtc.bitdepth(bits=8,dmode=dither)
    elif isinstance(dither,str):
        rehtid=lambda x:zvs.simplebitdepth(x,8,dither)
    elif callable(dither):
        rehtid=dither
    if count==None:
        count=2**depth
    scaling=255/(count-1)
    scalingc=scaling if not cs2 else scaling/2
    cs=cs or cs2
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

def naive32k(src,shift=0):
    clip=core.std.SeparateFields(src,tff=1)
    _tf=clip[::2]
    _bf=clip[1::2]
    match shift:
        case 0:
            selector_tf=[0,1,1,2,3] #1
            selector_bf=[0,1,2,3,3] #3
        case 1:
            selector_tf=[0,1,2,2,3] #2
            selector_bf=[0,0,1,2,3] #0
        case 2:
            selector_tf=[0,1,2,3,3] #3
            selector_bf=[0,1,1,2,3] #1
        case 3:
            selector_tf=[0,0,1,2,3] #0
            selector_bf=[0,1,2,2,3] #2
    _tf=_tf.std.SelectEvery(4,selector_tf)
    _bf=_bf.std.SelectEvery(4,selector_bf)
    clip=core.std.Interleave([_tf,_bf])
    clip=core.std.DoubleWeave(clip)[::2]
    clip=clip.std.SetFieldBased(2)
    return clip

def naive32i(src,shift=0):
    clip=core.std.SeparateFields(src,tff=1)
    _tf=clip[::2]
    _bf=clip[1::2]
    match shift:
        case 0:
            selector_tf=[0,1,3,4] #2
            selector_bf=[0,1,2,3] #4
        case 1:
            selector_tf=[0,1,2,4] #3
            selector_bf=[1,2,3,4] #0
        case 2:
            selector_tf=[0,1,2,3] #4
            selector_bf=[0,2,3,4] #1
        case 3:
            selector_tf=[1,2,3,4] #0
            selector_bf=[0,1,3,4] #2
        case 4:
            selector_tf=[0,2,3,4] #1
            selector_bf=[0,1,2,4] #3
    _tf=_tf.std.SelectEvery(5,selector_tf)
    _bf=_bf.std.SelectEvery(5,selector_bf)
    clip=core.std.Interleave([_tf,_bf])
    clip=core.std.DoubleWeave(clip)[::2]
    clip=clip.std.SetFieldBased(0)
    return clip

# the grey from black and white
# prepare your frame props btw, im relying on auto selection of core.resize
def grey(src,fp32=1,matrix='709',linearize=1):
    if src.format.color_family==vs.GRAY:
        return src
    elif src.format.color_family==vs.YUV:
        clip=core.resize.Spline64(src,format=vs.RGBS if fp32 else vs.RGB48)
    else:
        clip=zvs.sb(src,32 if fp32 else 16)
    if isinstance(matrix,int) and matrix>=0 and matrix<256:
        matrixval=matrix
    else:
        matrixvaldict={'rgb':0,'709':1,'fcc':4,'470bg':5,'170m':6,'240m':7,'ycgco':8,'2020ncl':9,'2020cl':10,'ydzdx':11,'601':5,709:1,601:5}
        matrixval=matrixvaldict.get(matrix)
    if matrixval==None: raise ValueError
    if linearize: clip=core.resize.Point(clip,transfer_s='linear')
    if matrixval==11:
        clip=core.fmtc.matrix(clip,mats='rgb',matd='ydzdx')
        clip=core.std.ShufflePlanes(clip,0,vs.GRAY)
        clip=zvs.setmatrix(clip,1)
    else:
        clip=core.resize.Point(clip,format=vs.GRAYS if fp32 else vs.GRAY16,matrix=matrixval)
    if linearize: clip=core.resize.Point(clip,transfer=src.get_frame(0).props._Transfer)
    return clip

# "lens blur"
def lamb(clip,radius=10,shape=None,ss=True):
    if shape:
        import shapely
        p=shapely.Polygon(shape)
    n=33 #acts like an upper limit
    o=int(n/2)
    def meow(sx=1,sy=1): #this is what lambs sound like
        w=0
        e=''
        for i in range(n):
            for j in range(n):
                x=j-o
                y=i-o
                xs=x*sx
                ys=y*sy
                if shape:
                    pt=shapely.Point(xs,ys)
                    if pt.within(p) or pt.touches(p):
                        e+=f" x[{x},{y}]"
                        w+=1
                else:
                    if (xs*xs)+(ys*ys) <= radius**2:
                        e+=f" x[{x},{y}]"
                        w+=1
        e+=" +" * (w-1)
        e+=f" {w} /"
        return e
    if clip.format.color_family==vs.YUV and ss:
        return core.akarin.Expr(clip,[meow(),meow(2**clip.format.subsampling_w,2**clip.format.subsampling_h)])
    else:
        return core.akarin.Expr(clip,meow())

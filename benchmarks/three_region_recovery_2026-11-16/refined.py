"""Tighter interval evaluation; the traffic model and proof are unchanged."""
from mfd_multiregion_transfer import *

def speed_range(total):
    # Range of a convex quadratic: maximum at an endpoint, minimum at
    # an endpoint or the vertex. Evaluate every candidate outward.
    a,b,c=I(MFD_A),I(MFD_B),I(MFD_C)
    lo,hi=I(float(total.lo)),I(float(total.hi))
    left=(a*lo+b)*lo+c;right=(a*hi+b)*hi+c
    vertex= -iv.mpf(MFD_B)/(2*iv.mpf(MFD_A))
    vertex_interval=I(lower(vertex.a),upper(vertex.b))
    values=[left,right]
    if float(total.lo)<=vertex_interval.hi and vertex_interval.lo<=float(total.hi):
        values.append((a*vertex_interval+b)*vertex_interval+c)
    return I(min(float(x.lo) for x in values),max(float(left.hi),float(right.hi)))

def traffic_jacobian(net,box,command):
    H=I(np.zeros((net.nc,net.nc)),np.zeros((net.nc,net.nc)))
    for i in range(net.r):
        total=isum(box[i*net.r+j] for j in range(net.r))
        speed=speed_range(total)
        slope=total*I(2.)*I(MFD_A)+I(MFD_B)
        for j in range(net.r):
            row=i*net.r+j
            for k in range(net.r):
                col=i*net.r+k
                component=box[row]
                others=isum(box[i*net.r+q] for q in range(net.r) if q!=j)
                # The Jacobian entry is a quadratic in this destination count
                # and the sum of the other counts. If both partial derivatives
                # have a fixed sign over the entire box, its extrema occur at
                # the corresponding corners. Otherwise retain the interval
                # expression. This tightens evaluation, not the admissible set.
                grads=[(slope*I(2.) if j==k else slope)+I(2.)*I(MFD_A)*component,
                       (slope if j==k else I(0.))+I(2.)*I(MFD_A)*component]
                boxes=[component,others]
                if all(g.lo>=0 or g.hi<=0 for g in grads):
                    extrema=[]
                    for maximum in [False,True]:
                        vals=[I(float(b.hi if (g.lo>=0)==maximum else b.lo)) for b,g in zip(boxes,grads)]
                        count,other=vals;count_sum=count+other
                        value=count*(I(2.)*I(MFD_A)*count_sum+I(MFD_B))
                        if j==k:value=value+(I(MFD_A)*count_sum+I(MFD_B))*count_sum+I(MFD_C)
                        extrema.append(value)
                    deriv=I(float(extrema[0].lo),float(extrema[1].hi))
                else:
                    deriv=component*slope+(speed if j==k else I(0.))
                deriv=deriv*I(1. if i==j else command[i])
                value=H[row,col]-deriv
                H.lo[row,col],H.hi[row,col]=value.lo,value.hi
                if i!=j:
                    target=((i+1)%net.r)*net.r+j
                    value=H[target,col]+deriv
                    H.lo[target,col],H.hi[target,col]=value.lo,value.hi
    return H

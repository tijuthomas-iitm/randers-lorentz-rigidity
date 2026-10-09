# Flag curvature under a gauge shift: F=|y| vs F=|y|+2k x1 y1 on the Euclidean plane (manuscript Remark 2.4, Appendix A.6).
# Expected: K(|y|)=0; K(shifted)=3 k^2 y1^4/F^4 (0.00340740404223 and 0.0485135314845 at k=1/5).
import sympy as sp
x=sp.symbols('x0 x1'); y=sp.symbols('y0 y1'); k=sp.symbols('k')
def flag(F,pt,yv,uv):
    n=2; E=F**2/2
    g=sp.Matrix(n,n,lambda i,j: sp.diff(E,y[i],y[j])); gi=g.inv()
    rhs=[sum(sp.diff(E,x[a],y[l])*y[a] for a in range(n))-sp.diff(E,x[l]) for l in range(n)]
    G=[sum(gi[i,l]*rhs[l] for l in range(n))/2 for i in range(n)]
    R=sp.Matrix(n,n,lambda i,c: 2*sp.diff(G[i],x[c])-sum(y[j]*sp.diff(G[i],x[j],y[c]) for j in range(n))
        +2*sum(G[j]*sp.diff(G[i],y[j],y[c]) for j in range(n))-sum(sp.diff(G[i],y[j])*sp.diff(G[j],y[c]) for j in range(n)))
    s={**dict(zip(x,pt)),**dict(zip(y,yv)),k:sp.Rational(1,5)}
    gn=g.subs(s).evalf(30); Rn=R.subs(s).evalf(30); u=sp.Matrix(uv); yy=sp.Matrix(yv)
    num=(u.T*gn*Rn*u)[0]; den=(yy.T*gn*yy)[0]*(u.T*gn*u)[0]-((yy.T*gn*u)[0])**2
    return num/den
alpha=sp.sqrt(y[0]**2+y[1]**2)
pt=[sp.Rational(1,3),sp.Rational(1,2)]
for yv in ([1,sp.Rational(1,2)],[sp.Rational(1,3),1]):
  u=[-yv[1],yv[0]]
  K0=flag(alpha,pt,yv,u); K1=flag(alpha+2*k*x[1]*y[1],pt,yv,u)
  F=(alpha+2*k*x[1]*y[1]).subs({**dict(zip(y,yv)),**dict(zip(x,pt)),k:sp.Rational(1,5)}).evalf(30)
  print("y",yv,"K(alpha)=",sp.N(K0,12),"K(alpha+dchi)=",sp.N(K1,12),"  3k^2 y1^4/F^4=",sp.N(3*sp.Rational(1,25)*sp.Rational(yv[1])**4/F**4,12))

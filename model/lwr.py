import numpy as np
from bokeh.plotting import figure,show
from bokeh.layouts import gridplot

def radial(x,x0,tau):
    return np.exp(-np.sum((x-x0)**2,axis=1)/(2*tau**2))

def logical_regression(x,x0,y,tau):
    x0=np.r_[1,x0]
    x_=np.c_[np.ones(len(x)),x]
    w=radial(x_,x0,tau)
    XTW=x_.T*w
    beta=np.linalg.pinv(XTW @ x_) @ XTW @ y

n=1000
x=np.linspace(-3,3,n)
y=np.log(np.abs(x**2-1)+0.5)
x+=np.random.normal(0,0.1,size=n)
domain=np.linspace(-3,3,300)

def lwr_plot(tau):
    preds=[logical_regression(x,x0,y,tau) for x0 in domain]
    p=figure(title=f"tau={tau}",width=400,height=400)
    p.scatter(x,y,alpha=0.3)
    p.line(domain,preds,line_width=0.4,color='red')


show(gridplot(
    [
        [lwr_plot(10),lwr_plot(11)],
        [lwr_plot(0.1),lwr_plot(0.01)]
    ]))



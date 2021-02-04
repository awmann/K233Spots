PRO doublespectra

modelpath = '/Users/andrewmann/Dropbox/Radii/Models_CIFIST_Dupuy.fits'
models = mrdfits(modelpath,1,header)
modelspectra = models.spectrum
modelheader = models.header
lambda_m = sxpar(modelheader,'LAMBDA_0') + findgen(sxpar(modelheader,'NLAMBDA'))*sxpar(modelheader,'D_LAMBDA')
teff = models.teff
logg = abs(models.logg)
metal = models.metal
afe = models.a_fe

l = where(teff eq 2800 and logg eq 5 and metal eq 0)
coolspec = modelspectra[l,*]
l = where(teff eq 3500 and logg eq 5 and metal eq 0 and afe eq 0)
hotspec = modelspectra[l,*]
l = where(teff eq 3300 and logg eq 5 and metal eq 0 and afe eq 0)
guessspec = modelspectra[l,*]

xrange1 = [5300,8500]
xrange2 = [10000,25000]
!p.multi=[0,1,2]
;plot,lambda_m,hotspec,/xstyle,/ystyle,xrange=xrange
;plot,lambda_m,coolspec,/xstyle,/ystyle,xrange=xrange

combined = (hotspec*0.27+0.73*coolspec)
combined/=median(combined[where(lambda_m gt xrange1[0] and lambda_m lt xrange1[1])])
hotspec/=median(hotspec[where(lambda_m gt xrange1[0] and lambda_m lt xrange1[1])])
guessspec/=median(guessspec[where(lambda_m gt xrange1[0] and lambda_m lt xrange1[1])])
plot,lambda_m,hotspec,/xstyle,/ystyle,xrange=xrange1,yrange=[0.4,2.0]
oplot,lambda_m,combined,color=cgcolor('red')
oplot,lambda_m,guessspec,color=cgcolor('green')

combined = (hotspec*0.27+0.73*coolspec)
combined/=median(combined[where(lambda_m gt xrange2[0] and lambda_m lt xrange2[1])])
hotspec/=median(hotspec[where(lambda_m gt xrange2[0] and lambda_m lt xrange2[1])])
guessspec/=median(guessspec[where(lambda_m gt xrange2[0] and lambda_m lt xrange2[1])])
plot,lambda_m,hotspec,/xstyle,/ystyle,xrange=xrange2
oplot,lambda_m,combined,color=cgcolor('red')
oplot,lambda_m,guessspec,color=cgcolor('green')

END


estnoisesrcs := $(wildcard $(srcdir)/estnoise/*.c)
estnoiseobjs := $(estnoisesrcs:.c=.o)

.INTERMEDIATE: $(estnoiseobjs)

lib/libestnoise.a: libestnoise.a($(estnoiseobjs))

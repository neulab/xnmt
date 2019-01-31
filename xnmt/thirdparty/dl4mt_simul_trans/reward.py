# Modified by: philip30
# Source repo: https://github.com/nyu-dl/dl4mt-simul-trans/blob/master/reward.py


import numpy
from xnmt.thirdparty.dl4mt_simul_trans.bleu import SmoothingFunction, sentence_bleu

# These values are looked up from config.py
cw = 8
target = 0.8

def BLEUwithForget(words, ref, act):
  q0  = numpy.zeros((len(act),))

  # check 0, 1
  maps  = [(it, a) for it, a in enumerate(act) if a < 2]
  kmap  = len(maps)
  q   = numpy.zeros((kmap,))

  chencherry = SmoothingFunction()

  # compute BLEU for each Yt
  Y = []
  bleus = []
  truebleus = []

  if len(words) == 0:
    bleus = [0]
    truebleus = [0]

  ref = [ref[i] for i in range(ref.len_unpadded())]
  for t in range(len(words)):
    Y.append(words[t])
    bb = sentence_bleu([ref], Y, smoothing_function=chencherry.method5)

    bleus.append(bb[1])   # try true BLEU
    truebleus.append(bb[1])
  
  # print 'Latency BLEU', lbn
  bleus = [0] + bleus  # use TRUE BLEU
  bleus = numpy.array(bleus)
  temp  = bleus[1:] - bleus[:-1]

  tpos  = 0
  for pos, (it, a) in enumerate(maps):
    if (a == 1) and (tpos < len(words)):
      q[pos] = temp[tpos]
      q0[it] = q[pos]
      tpos  += 1

  # add the whole sentence balance on it
  q0[-1] = truebleus[-1]  # the last BLEU we use the real BLEU score.
  return q0

def NormalizedDelay(act, source_len):
  d = numpy.zeros((len(act),))
  # print a
  _src = 0
  _trg = 0
  _sum = 0
  for it, a in enumerate(act):
    if a == 0:
      if _src < source_len:
        _src += 1
    elif a == 1:
      _trg += 1
      _sum += _src
  d[-1] = _sum / (_src * _trg + 1e-6)
  return d

def MaximumDelay2(act, _max=5, beta=0.1):
  d  = numpy.zeros((len(act),))
  _cur = 0
  for it, a in enumerate(act):
    if a == 0:
      _cur += 1
      if _cur > _max:
        d[it] = -0.1 * (_cur - _max)
      pass
    elif a == 1:   # only for new commit
      _cur = 0

  return d * beta

# The general function for rewards (for simultrans):
def return_reward(words, ref, act, source_len):
  beta   = 0.03 # 0.5

  q0 = BLEUwithForget(words, ref, act)
  d0 = NormalizedDelay(act, source_len)
  # global reward signal :::>>>
  # just bleu
  bleu  = q0[-1]

  # just delay
  delay = d0[-1]

  # local reward signal :::>>>>
  # use maximum-delay + latency bleu (with final BLEU)
  q = q0
  q[-1] = 0
  if cw > 0:
    d = MaximumDelay2(act, _max=cw, beta=beta)
  else:
    d = 0

  r0  = q + 0.5 * d

  if target < 1:
    tar = -numpy.maximum(delay - target, 0)
  else:
    tar = 0

  rg  = bleu + tar # it is a global reward, will not be discounted.
  r    = r0
  r[-1] += rg

  R = r[::-1].cumsum()[::-1]
  return R, bleu, delay, R[0]


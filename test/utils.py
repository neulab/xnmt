def has_cython():
  try:
    from xnmt.cython import xnmt_cython
    return True
  except:
    return False
  
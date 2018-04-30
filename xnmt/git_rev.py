def get_git_revision():
  from subprocess import CalledProcessError, check_output
  try:
    command = 'git rev-parse --short HEAD'
    print("checking git revision in", __file__)
    rev = check_output(command.split(u' '), cwd=os.path.dirname(__file__)).decode('ascii').strip()
  except (CalledProcessError, OSError):
    rev = None
  return rev
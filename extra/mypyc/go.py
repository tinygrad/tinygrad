from weak import Hello, create_ref
class Hello2(object):
  def __init__(self):
    self.a = "bob"
h = Hello()
print(h)
r = create_ref(h)
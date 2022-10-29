class A:
    def __init__(self):
        self.me = None
        self.param = 14

    def get_or_create_param(self):
        if not 'param' in self.__dict__:
            self.param = 12


a = A()
a.get_or_create_param()
print(a.__dict__)

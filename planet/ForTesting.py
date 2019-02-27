
class Animal():
    def __init__(self):
        self.age = 1

class Cat(Animal):
    def __init__(self):
        super().__init__()
        self.age = 2
        self.name = 'Cat'


c = Cat()
print(c.age)
print(c.name)
i = "hello! I love python class I like to study."

newList = i.split()
print("newList\n", newList)
newList = i.split('!')
print("newList\n", newList)
newList = i.split(maxsplit=2)
print("newList\n", newList)
newList = i.split(maxsplit=3)
print("newList\n", newList)
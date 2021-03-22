import sys

if (len(sys.argv) != 4):
    print("Wrong parameters. Usage: %s <file-path> -[bl] <busId-lineId>" % (sys.argv[0]))
    exit(-1)

class Route:
    def __init__(self, busId, routeId, x, y, seconds):
        self.busId =  busId
        self.routeId = routeId
        self.x = x
        self.y = y
        self.seconds = seconds

    def computeDistance(self):
        return (self.x**2 + self.y**2) ** 0.5

f = open(sys.argv[1], "r")
if (f == None):
    print("Can't open database file.")

tot = 0
n = 0
for line in f:
    busId, routeId, x, y, seconds = line.split()
    r = Route(busId, routeId, int(x), int(y), int(seconds))
    if (sys.argv[2] == "-b"):
        if (r.busId == sys.argv[3]):
            tot += r.computeDistance()
    elif (sys.argv[2] == "-l"):
        if (r.routeId == sys.argv[3]):
            tot += r.seconds
            n+=1

if (sys.argv[2] == "-b"):
    print("%s Total Distance: %d" % (sys.argv[3], tot))
elif (sys.argv[2] == "-l"):
    print("%s - Avg Speed: %f", sys.argv[3], tot/n)

f.close()
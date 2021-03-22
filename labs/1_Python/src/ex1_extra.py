import sys

FPATH = "./Data/ex1_extra_data.txt"
MONTHS = [ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

fp = open(FPATH, "r")
birthCity = {}
birthMonth = [0 for i in range(13)]
births=0
for line in fp:
    name, surname, bplace, bdate = line.split()
    #print(name,surname,bplace,bdate)
    # Increment birth for each city
    if (bplace not in birthCity):
        birthCity[bplace] = 0
    birthCity[bplace] += 1

    # Increment birth for each month
    month = int(bdate.split('/')[1])
    birthMonth[month] += 1

    births +=1

fp.close()

# Compute desired statistics
birthsPerCity = births / len(birthCity)

# Printing
print("Births per city:")
for city in birthCity:
    print("%s: %d" % (city, birthCity[city]))

print("Births per month:")
for id, count in enumerate(birthMonth[1:]):
    if (count != 0):
        print("%s: %d" % (MONTHS[id], count))

print("Average number of births per city: %.2f" % (birthsPerCity))
#CarEvaluation class with constructor and showEvaluation method
class CarEvaluation:
    carCount = 0
    def __init__(self, Brand, Price, Safety):
        CarEvaluation.carCount = CarEvaluation.carCount + 1 #increment counter every time method is called
        self.brand = Brand #assign variables
        self.price = Price
        self.safety = Safety
        self.carCount = CarEvaluation.carCount

    def showEvaluation(self): #print car data
        print "The ", self.brand, " has a ", self.price, " price and it's safety is rated a ", self.safety

def sortbyprice(arr, order): #print brands by price in order defined below
    a = []
    for i, val in enumerate(arr.price):
        a.append(arr[i].brand)
    if order == 'asc':
        a.reverse()
    print(a)

def searchforsafety(arr, val): #search for safety values
    a = []
    for i in range(0,len(arr)):
        a.append(arr[i].safety)
    if val in a:
        return True
    else:
        return False

if __name__ == "__main__":
    eval1 = CarEvaluation("Ford", "High", 2)
    eval2 = CarEvaluation("GMC", "Med", 4)
    eval3 = CarEvaluation("Toyota", "Low", 3)

    print("Car Count = %d" % CarEvaluation.carCount) # Car Count = 3

    eval1.showEvaluation() #The Ford has a High price and it's safety is rated a 2
    eval2.showEvaluation() #The GMC has a Med price and it's safety is rated a 4
    eval3.showEvaluation() #The Toyota has a Low price and it's safety is rated a 3

    L = [eval1, eval2, eval3]

    sortbyprice(L, "asc"); #[Toyota, GMC, Ford]
    sortbyprice(L, "des"); #[Ford, GMC, Toyota]
    print searchforsafety(L, 2); #true
    print searchforsafety(L, 1); #false
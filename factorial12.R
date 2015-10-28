#using for loop
a = 12
f1 = 1
for(i in 1:12)
{
  f1 = f1*a
  a = a-1
  
}
cat("12! = ", f1)

#using while loop
b = 12
f2 = 1
while (b!=0)
{
  f2 = f2*b
  b = b-1
}
cat("12! = ", f2)

#question 2
count = 1
for (i in 20:50)
{
  if (i%%5 == 0)
  {
    vec1[count] = i
  count = count +1
  }
  
}
vec1

#question 3


a1 <- readline("What is the value of a? ");
b <- readline("What is the value of b? ");
c <- readline("What is the value of c? ");
a1 = strtoi(a1)
b = strtoi(b)
c = strtoi(c)
x1 <- ((-b) + sqrt(b^2-4*c*a1))/(2*a1);
x2 <- ((-b) - sqrt(b^2-4*c*a1))/(2*a1);

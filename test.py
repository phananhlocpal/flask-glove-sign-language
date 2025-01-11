a = int(input("Nhap so a: "))
b = int(input("Nhap so b: "))
c = int(input("Nhap so c: "))

def kiem_tra_tam_giac(a, b, c):
    if a + b > c and a + c > b and b + c > a:
        return True
    return False

print(kiem_tra_tam_giac(a, b, c))
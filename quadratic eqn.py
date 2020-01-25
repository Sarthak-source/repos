import math

from tkinter import *

from PIL import Image, ImageTk

from tkinter.font import Font


  


root=Tk()
root.title('Quadratic equation solver')

canvas=Canvas(root,width=110,height=110,bg='black',highlightthickness=0)
image1=ImageTk.PhotoImage(Image.open('D:\phython assignment\quadratic\shoot.ico'))
canvas.create_image(0,0,anchor=NW,image=image1)
canvas.pack()

n_font = Font(family='Helvetica', size=16)
heading=Label(root,text='QUADRATIC EQUATION SOLVER',fg='white',bg='black',font=n_font)
heading.pack()


m_font = Font(family='Helvetica', size=10)



root.configure(bg='black')
root.iconbitmap(r'D:\phython assignment\quadratic\shoot.ico')



a1=Label(root,text='Enter value of a',fg='white',bg='black',font=m_font)
a1.pack()

aval= Entry(root)
aval.pack()



b1=Label(root,text='Enter value of b',fg='white',bg='black',font=m_font)
b1.pack()

bval= Entry(root)
bval.pack()



c1=Label(root,text='Enter value of c',fg='white',bg='black',font=m_font)
c1.pack()

cval= Entry(root)
cval.pack()




def baska():


    a=int(aval.get())
    b=int(bval.get())
    c=int(cval.get())

    

    s=(b*b)-4*a*c

    if(s<0):
        
        i='Root is imaginary'
        im=Label(root,text=i,fg='white',bg='black',font=m_font)
        im.pack()

        
    else:


        ri=Label(root,text='Roots are real',fg='white',bg='black',font=m_font)
        ri.pack()
        

        x_1=(-b+math.sqrt(s))/(2*a)
        x_2=(-b-math.sqrt(s))/(2*a)

        rl=Label(root,text=x_1,fg='white',bg='black',font=m_font)
        rl.pack()
        

        rl2=Label(root,text=x_2,fg='white',bg='black',font=m_font)
        rl2.pack()
    

newbutton=Button(root,text='Calculate',command=baska,fg='white',bg='black',font=m_font)
newbutton.pack()


root.mainloop()

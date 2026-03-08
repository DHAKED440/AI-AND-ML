Library_data={
 "harry potter and the sorcerer's stone"  :  "J.K. Rowling" ,        
 "the alchemist"                          : "Paulo Coelho" ,        
 "atomic habits"                          :  "James Clear" ,         
 "rich dad poor dad"                      :  "Robert T. Kiyosaki" ,  
 "think and grow rich"                      : "Napoleon Hill"       ,
 "the power of now"                         : "Eckhart Tolle"       ,
 "to kill a mockingbird"                    : "Harper Lee"         ,
 "1984"                                     : "George Orwell"       ,
 "pride and prejudice"                      : "Jane Austen"         ,
 "the great gatsby"                         : "F. Scott Fitzgerald" ,
 "the hobbit"                               : "J.R.R. Tolkien"      ,
 "the catcher in the rye"                   : "J.D. Salinger"        ,
 "the da vinci code"                        : "Dan Brown"  ,         
 "the 7 habits of highly effective people"  : "Stephen R. Covey" ,   
 "the subtle art of not giving a fuck"      : "Mark Manson" ,        
 "the psychology of money"                  : "Morgan Housel" ,      
 "sapiens: a brief history of humankind"    : "Yuval Noah Harari",   
 "the kite runner"                          : "Khaled Hosseini" ,    
 "the book thief"                         : "Markus Zusak"        

}

borrowed={}
while True:
    print("\n------ LIBRARY MANAGER ------")
    print("1. Search Book")
    print("2. Return Book")
    print("3. Check Availability")
    print("4. Exit")

    choice=input("Enter your choice: ")

    if choice=='1' :
        title=input("Enter the book name : ").lower()
        if title in Library_data :

            print("Author : " , Library_data[title])
            book=input("Do You Want This Book ? (Yes/No) : ").lower()

            if book == 'yes' :
              
              borrowed[title]=Library_data[title]
              del Library_data[title]
              print("Book is succesfully borrowed")
            elif book == "no":
              print("Thank you for using Library Manager")
              break
            else:
              print("Please enter only yes or no")    

        elif title in borrowed :
            print("Book is in borrowed list")
        else : 
            print("No such book exist in Library")

    #return Status
    elif choice=='2' :
        title=input("Enter the book name to return : ").lower()

        if title in borrowed :
          Library_data[title]=borrowed[title]
          del borrowed[title]
          print("Book is return succesfully")
        else :
            print("book is not in borrowed") 


    elif choice =='3' :
        title=input("Enter the book name : ")

        if title in Library_data :
            print("Book is available")
        else :
            print("Book is not available")

    elif choice=="4":
        print("Thank you for using Library Manager")
        break

    else:
        print("Invalid choice")                                     
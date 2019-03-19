"""game=[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],]


def game_board(game_map, player=0, row=0 , column=0,just_display=False):
   try:
        print ("   0  1  2")
        if not just_display:
            game_map[row][column] = player

        for count, row in enumerate(game_map):
            print (count, row)
        return game_map
  # except IndexError as e:
  #      print ("Error : make sure you input row/column as 0 1 or 2?",e)
   except Exception as e:
        print("something went very wrong!", e)  


game = game_board(game,just_display=True) 
game = game_board(game,player=1, row=3, column=0  )
"""
game=[[2, 1, 1],
      [2, 2, 0],
      [2, 1, 2]]


for col in range(len(game)):
    check=[]
    for row in game:
        check.append(row[col])

    if check.count(check[0])== len(check) and check[0] !=0:
            

            print("Winner!")    

'''
def win(current_game):
    for row in game:
        print(row)
        print(row.count(row[0]))
        print(len(row))
        if row.count(row[0])== len(row) and row[0] !=0:
            

            print("Winner!")
        """col1=row[0]
        col2=row[1]
        col3=row[2]

        if col1 == col2 == col3:
            print("winner!!!")"""

win(game)'''
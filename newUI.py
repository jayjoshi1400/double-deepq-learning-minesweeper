import tkinter as tk
from tkinter import ttk

class TkinterRenderer:
    def __init__(self, board):
        self.board = board
        self.root = tk.Tk()
        self.root.title("Minesweeper")
        self.load_images()
        self.display()

    def load_images(self):
        self.images = {
            ' ': tk.PhotoImage(file="blank.png").subsample(4, 4),
            '1': tk.PhotoImage(file="one.png").subsample(4, 4),
            '2': tk.PhotoImage(file="two.png").subsample(4, 4),
            '3': tk.PhotoImage(file="three.png").subsample(4, 4),
            '4': tk.PhotoImage(file="four.png").subsample(4, 4),
            '5': tk.PhotoImage(file="5.png").subsample(4, 4),
            '6': tk.PhotoImage(file="6.png").subsample(4, 4),
            '7': tk.PhotoImage(file="seven.png").subsample(4, 4),
            '8': tk.PhotoImage(file="four.png").subsample(4, 4),
            'F': tk.PhotoImage(file="flag.png").subsample(4, 4),
        }

    def display(self):
        self.frm = ttk.Frame(self.root, padding=20)
        self.frm.grid()
        self.labels = {}

        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                tile = self.board[row][col]
                label = ttk.Label(self.frm, image=self.images[tile])
                label.grid(column=col, row=row)
                self.labels[(row, col)] = label

        ttk.Button(self.frm, text="Quit", command=self.root.destroy).grid(column=len(self.board[0]), row=len(self.board))

    def update(self, new_board):
        # print('here')
        self.board = new_board
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                tile = self.board[row][col]
                self.labels[(row, col)].config(image=self.images[tile])
        # Keep the window responsive
        self.root.update_idletasks()
        self.root.update()

import json

class FEN:
    fen = ''
    pieces = None

    def __init__(self, filename):
        self.filename = filename
        self.fen = self.readFENfromFile()
        self.pieces = FEN.fen_to_dict(self.fen)

    
    def readFENfromFile(self):
        with open("./FEN/" + self.filename + ".json", 'r') as jsonfile:
            jsondata = json.load(jsonfile)
        return jsondata['fen']


    @staticmethod
    def fen_to_dict(fen):
        num_to_coord = {i: f"{chr(ord('A') + i % 8)}{8 - i // 8}" for i in range(64)}
        board_dict = {}
        piece_placement = fen.split(' ')[0]

        fen_to_piece = {
            'K': 'w_King', 'Q': 'w_Queen', 'R': 'w_Rook', 'B': 'w_Bishop', 'N': 'w_Knight', 'P': 'w_Pawn',
            'k': 'b_King', 'q': 'b_Queen', 'r': 'b_Rook', 'b': 'b_Bishop', 'n': 'b_Knight', 'p': 'b_Pawn'
        }

        # i [0, 63], index of the square
        i = 0
        for char in piece_placement:
            if char.isnumeric():
                pass
                i += int(char)
            elif char.isalpha():
                square = num_to_coord[i]
                piece = fen_to_piece[char]
                board_dict[square] = piece
                i += 1

        return board_dict

    @staticmethod
    def dict_to_fen(board_dict):
        coord_to_num = {f"{chr(ord('A') + i % 8)}{8 - i // 8}": i for i in range(64)}
        fen = ''
        piece_to_fen = {
            'w_King': 'K', 'w_Queen': 'Q', 'w_Rook': 'R', 'w_Bishop': 'B', 'w_Knight': 'N', 'w_Pawn': 'P',
            'b_King': 'k', 'b_Queen': 'q', 'b_Rook': 'r', 'b_Bishop': 'b', 'b_Knight': 'n', 'b_Pawn': 'p'
        }

        # i number of square analyzed
        i = 0
        empty_square_count = 0
        for square in sorted(coord_to_num.keys(), key=lambda x: coord_to_num[x]):
            if square in board_dict:
                if empty_square_count > 0:
                    fen += str(empty_square_count)
                    empty_square_count = 0
                piece = board_dict[square]
                fen += piece_to_fen[piece] if piece else '1'
            else:
                empty_square_count += 1

            i += 1

            if i % 8 == 0 and i <= 64:
                if empty_square_count > 0:
                    fen += str(empty_square_count)
                    empty_square_count = 0
                if i<64:
                    fen += '/'

        return fen


# Tests FEN
if True:
    fen_notations_test = (
        "8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1",
    #    "r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1",
    #    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    )

    for fen_notation in fen_notations_test:
        print("####### fen_to_dict #######")
        chess_board = FEN.fen_to_dict(fen_notation)
        print(fen_notation + " parsed to:\n" + str(chess_board))

        print("\n####### dict_to_fen #######")
        fen_notation_output = FEN.dict_to_fen(chess_board)
        print(fen_notation + " parsed and refened to: " + fen_notation_output)
        print(fen_notation.split(' ')[0] == fen_notation_output)
        print()


#Test JSON
if True:
    filename = "0002"
    print("image " + filename + " position:")
    print(FEN(filename).fen)
    print(FEN(filename).pieces)

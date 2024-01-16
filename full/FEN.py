import json
import chess


class FEN:
    fen = ''
    pieces = None
    fen_to_piece = {
                'K': 'w_King', 'Q': 'w_Queen', 'R': 'w_Rook', 'B': 'w_Bishop', 'N': 'w_Knight', 'P': 'w_Pawn',
                'k': 'b_King', 'q': 'b_Queen', 'r': 'b_Rook', 'b': 'b_Bishop', 'n': 'b_Knight', 'p': 'b_Pawn'
            }
        
    def __init__(self, filename):
        self.filename = filename
        self.fen, white_view = self._parseJSON()
        self.pieces = FEN.fen_to_dict(self.fen)
        
        if white_view:
            self.view = "white"
        else:
            self.view = "black"
    
    
    
    def _parseJSON(self):
        with open(self.filename + ".json", 'r') as jsonfile:
            jsondata = json.load(jsonfile)
        return jsondata['fen'], jsondata['white_turn']


    def print(self):
        board = chess.Board(self.fen)
        print(board)

    @staticmethod
    def fen_to_dict(fen):
        num_to_coord = {i: f"{chr(ord('A') + i % 8)}{8 - i // 8}" for i in range(64)}
        board_dict = {}
        piece_placement = fen.split(' ')[0]

        # i [0, 63], index of the square
        i = 0
        for char in piece_placement:
            if char.isnumeric():
                pass
                i += int(char)
            elif char.isalpha():
                square = num_to_coord[i]
                piece = FEN.fen_to_piece[char]
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
    
    @staticmethod
    def print_from_FEN(fen : str):
        print(chess.Board(fen))
    
    @staticmethod
    def cmpFEN(fen1 : str, fen2 : str):
        board1 = chess.Board(fen1)
        board2 = chess.Board(fen2)
        
        diff_dict = {}
        
        for square in chess.SQUARES:
            if board1.piece_at(square) != board2.piece_at(square):
                piece1 = board1.piece_at(square)
                piece2 = board2.piece_at(square)
                
                if piece1 is not None:
                    piece1 = FEN.fen_to_piece[piece1.symbol()]
                else:
                    piece1 = 'empty'
                if piece2 is not None:
                    piece2 = FEN.fen_to_piece[piece2.symbol()]
                else:
                    piece2 = 'empty'
                
                diff_dict[chess.square_name(square)] = (piece1, piece2)
                
        return diff_dict
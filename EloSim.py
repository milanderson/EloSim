import pickle
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

PUB_ELOS = []
UNCERTAINTY = 32
LEARNING_RATE = 0.9

def main():
    players = init_players()
    playtournament(players, UNCERTAINTY, 10, False, 1, 1)
    players.sort()
    print("done")

class Player:
    def __init__(self, elo_tru, elo_est=1690, gr=0.0):
        self.elo_tru = elo_tru
        self.gr = gr
        self.id = len(PUB_ELOS)
        PUB_ELOS.append(elo_est)

    def __lt__(self, other):
        return PUB_ELOS[self.id] < PUB_ELOS[other.id]

    def __le__(self, other):
        return PUB_ELOS[self.id] <= PUB_ELOS[other.id]

    def __gt__(self, other):
        return PUB_ELOS[self.id] > PUB_ELOS[other.id]

    def __ge__(self, other):
        return PUB_ELOS[self.id] >= PUB_ELOS[other.id]

    def __eq__(self, other):
        return PUB_ELOS[self.id] == PUB_ELOS[other.id]

    def get_elo(self):
        return PUB_ELOS[self.id]

    def update_elo(self, new_elo):
        PUB_ELOS[self.id] = new_elo

'''
        @baselist       the list of players
        @begin          the beginning index
        @end            the end index
    A class for in-place cuts of a larger list
'''
class PlayerBracket(object):
    def __init__(self, baselist, begin, end=None):
        self._base = baselist
        self._begin = begin
        self._end = len(baselist) if end is None else end

    def __len__(self):
        return self._end - self._begin

    def __getitem__(self, i):
        return self._base[self._begin + i]

    def __setitem__(self, i, val):
        self._base[i + self._begin] = val

'''
        @players            the list of players
        @uncertainty        the adjustment rate. equivalent to the amount of points won/lost per match
        @rounds             the number of matches to play
        @userbrackets       the number of brackets (players match players only within their bracket)
        @show_elo_every     how often to draw the estimated Elo distribution. If None, nothing is shown.
        @show_dist_every    how often to draw the estimated versus real Elo. If None, nothing is shown.
    Matches all provided players, and adjusts their estimated scores based on match results
'''
def playtournament(players, uncertainty, rounds=20, usebrackets=True, show_elo_every=None, show_dist_every=None):
    if usebrackets:
        # calculate the bin sizes, round each bin to an even number of players
        hist, binedges = np.histogram(PUB_ELOS, bins=23)
    else:
        binedges = [min(PUB_ELOS), max(PUB_ELOS)]

    players.sort()
    bins = [0]
    edge = 1
    for i in range(len(players)):
        player = players[i]
        if edge < len(binedges) and PUB_ELOS[player.id] > binedges[edge]:
            bins.append(i)
            edge += 1
    bins.append(len(players))

    for j in range(rounds):
        print("Round " + str(j + 1))
        # pair players within their skill bracket, match them and record the new scores
        for i in range(len(bins) - 1):
            bracket = PlayerBracket(players, bins[i], bins[i+1])
            np.random.shuffle(bracket)
            pair = 0
            while pair < len(bracket) - 1:
                playmatch(bracket[pair], bracket[pair + 1], uncertainty)
                pair += 2
        
        if show_elo_every is not None and (j + 1) % show_elo_every == 0:
            plt.hist(np.array(PUB_ELOS), bins=23)
            plt.title("Round " + str(j + 1) + " Ratings")
            plt.xlabel("Elo")
            plt.ylabel("Frequency")

            plt.show()

        if show_dist_every is not None and (j + 1) % show_dist_every == 0:
            players.sort()
            tru_y_pos = np.arange(len(players))
            label_y_pos = [i*(len(players)/10) for i in range(10)]
            pub_elos = [str(int(PUB_ELOS[players[i].id])) for i in label_y_pos]
            tru_elos = [player.elo_tru for player in players]
            
            plt.bar(tru_y_pos, tru_elos, align='center')
            plt.xticks(label_y_pos, pub_elos)
            plt.xlabel('Measured Elo')
            plt.ylabel('True Elo')
            plt.title('Round ' + str(j + 1) + ' Elo Spread')
            
            plt.show()

'''
        @playerA            the first player in the match
        @playerB            the second player in the match
        @uncertainty        the weight of a win/loss
    Estimate the win chance of each player, flip a coin weighted on true skill, and update the scores based on the results
'''
def playmatch(playerA, playerB, uncertainty):
        # the logistic function for estimating win probability
        playerA_elo = PUB_ELOS[playerA.id]
        playerB_elo = PUB_ELOS[playerB.id]
        A_true_logit = float(playerB.elo_tru - playerA.elo_tru)/400.0
        A_pred_logit = float(playerB_elo - playerA_elo)/400.0
        A_pred_winchance = 1.0/(1.0 + 10.0**A_pred_logit)
        A_true_winchance = 1.0/(1.0 + 10.0**A_true_logit)

        if coinflip(A_true_winchance):
            PUB_ELOS[playerA.id] += uncertainty*(1 - A_pred_winchance)
            PUB_ELOS[playerB.id] += uncertainty*(A_pred_winchance - 1)
        else:
            PUB_ELOS[playerA.id] += uncertainty*(-A_pred_winchance)
            PUB_ELOS[playerB.id] += uncertainty*(A_pred_winchance)

def coinflip(win_chance=0.5):
    if np.random.uniform(0, 1) > win_chance:
        return False
    return True

def init_players(playercount=10000):
    f = open('data/chess_ratings.pkl','rb')
    ratings = pickle.load(f)

    # generate normal spread of scores
    gen_ratings = np.random.normal(loc=np.mean(ratings), scale=np.std(ratings), size=playercount) #np.array(ratings)

    # initialize the player base
    players = []
    for rating in gen_ratings:
        players.append(Player(rating))
    return players

if __name__=="__main__":
    main()
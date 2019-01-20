import pickle
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

PUB_ELOS = []
UNCERTAINTY = 32
LEARNING_RATE = 0.95

def main():
    players = init_players()
    play_tournament(players, team_size=1, uncertainty=UNCERTAINTY, learning_rate=1.0, rounds=10, usebrackets=False)

class Player:
    def __init__(self, elo_tru, elo_est=1690, gr=0.0):
        self.elo_tru = elo_tru
        self.gr = gr
        self.id = len(PUB_ELOS)
        self.rank_dists = [0,0,0,0,0,0,0,0,0,0]
        self.rank_i = 0
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

    def add_change(self, bracket_avg_pub_elo, bracket_avg_tru_elo, playerlist, uncertainty):
        # get elo distance from true rank
        raw_dist = PUB_ELOS[self.id] - PUB_ELOS[playerlist[self.id].id]
        
        # calculate win/loss chance
        pred_logit = float(bracket_avg_pub_elo - PUB_ELOS[self.id])/400
        true_logit = float(bracket_avg_tru_elo - self.elo_tru)/400
        if raw_dist > 0:
            pred_logit = float(PUB_ELOS[self.id] - bracket_avg_pub_elo)/400.0
            true_logit = float(self.elo_tru - bracket_avg_tru_elo)/400.0

        try:
            pred_winchance = 1.0/(1.0 + 10.0**pred_logit)
            true_winchance = 1.0/(1.0 + 10.0**true_logit)

            #calculate update amount for each game
            update_amount = 1 + uncertainty*(1 - pred_winchance)

            self.rank_dists[self.rank_i] = abs(raw_dist) / (true_winchance*update_amount)
            self.rank_i = (self.rank_i + 1) % 10
        except Exception as e:
            print(e, PUB_ELOS[self.id], bracket_avg_pub_elo, self.elo_tru, bracket_avg_tru_elo)

    def get_change_rate(self):
        return sum(self.rank_dists)/10.0

    def get_elo(self):
        return PUB_ELOS[self.id]

    def update_elo(self, new_elo):
        PUB_ELOS[self.id] = new_elo

'''
    A class for sorting in-place cuts of a larger list
        @baselist       the list of players
        @begin          the beginning index
        @end            the end index
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
    Matches all provided players, and adjusts their estimated scores based on match results
        @players            the list of players
        @team_size          the number of players on a team
        @uncertainty        the adjustment rate. equivalent to the amount of points won/lost per match
        @learning_rate      the rate of decay on the uncertainty
        @rounds             the number of matches to play
        @usebrackets        the number of brackets (players match players only within their bracket)
        @show_elo_every     how often to draw the estimated Elo distribution. If None, nothing is shown
        @show_dist_every    how often to draw the estimated versus real Elo. If None, nothing is shown
        @show_loss_every    how often to draw the average turns to correct rank. If None, nothing is shown
'''
def play_tournament(players, team_size = 1, uncertainty=32, learning_rate = 1.0, rounds=20, usebrackets=False, show_elo_every=None, show_dist_every=None, show_loss_every=None):

    poolsize = len(players)
    if usebrackets:
        poolsize /= 23

    for j in range(rounds):
        print("Round " + str(j + 1))
            
        playing_ppl = range(poolsize)
        matches_played = 0
        # pair players within their skill bracket, match them and record the new scores
        while 2*team_size*matches_played < len(players) - 1:
            match_pool = [playing_ppl.pop(0)]
            if playing_ppl[len(playing_ppl) - 1] < len(players) - 1:
                playing_ppl.append(playing_ppl[len(playing_ppl) - 1] + 1)

            for i in range((team_size*2) - 1):
                index = np.random.randint(len(playing_ppl))
                match_pool.append(playing_ppl.pop(index))

                if len(playing_ppl) > 0 and playing_ppl[len(playing_ppl) - 1] < len(players) - 1:
                    playing_ppl.append(playing_ppl[len(playing_ppl) - 1] + 1)

            teamA, teamB = maketeams([players[i] for i in match_pool])

            playmatch(teamA, teamB, uncertainty)
            matches_played += 1

        uncertainty = max(learning_rate*uncertainty, 10)
        binedges = update_player_change_rates(players, uncertainty)
        
        
        if show_elo_every is not None and (j + 1) % show_elo_every == 0:
            plt.hist(np.array(PUB_ELOS), bins=23)
            plt.title("Round " + str(j + 1) + " Ratings")
            plt.xlabel("Estimated Elo")
            plt.ylabel("Frequency")

            plt.show()

        if show_dist_every is not None and (j + 1) % show_dist_every == 0:
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

        if show_loss_every is not None and (j + 1) % show_loss_every == 0:

            bin_errors = [[]]
            items = [0]
            edge = 1
            for i in range(len(players)):
                if PUB_ELOS[players[i].id] < binedges[edge]:
                    bin_errors[edge-1].append(players[i].get_change_rate())
                    items[edge - 1] += 1
                else:
                    edge += 1
                    bin_errors.append([players[i].get_change_rate()])
                    items.append(1)
            
            #print(items, binedges)

            x = range(1, len(bin_errors) + 1)
            y = [np.mean(lst) for lst in bin_errors]
            yerr = [np.std(lst) for lst in bin_errors]
            plt.errorbar(x, y, yerr=yerr, fmt='bo-', ecolor='m', capthick=2)
            plt.xlabel('Skill Bracket')
            plt.ylabel('Avg Expected Matches to Rank')
            plt.title('Round ' + str(j + 1) + ' Error Rate')
            plt.show()

'''
    Estimate the win chance of each player, flip a coin weighted on true skill, and update the scores based on the results
        @playerA            the first player in the match
        @playerB            the second player in the match
        @uncertainty        the weight of a win/loss
'''
def playmatch(teamA, teamB, uncertainty):
        # uses the logistic function for estimating win probability
        teamA_elo_t = sum([player.elo_tru for player in teamA])
        teamB_elo_t = sum([player.elo_tru for player in teamB])
        teamA_elo_p = sum([PUB_ELOS[player.id] for player in teamA])
        teamB_elo_p = sum([PUB_ELOS[player.id] for player in teamB])

        A_true_logit = float(teamB_elo_t - teamA_elo_t)/400.0
        A_pred_logit = float(teamB_elo_p - teamA_elo_p)/400.0
        A_true_winchance = 1.0/(1.0 + 10.0**A_true_logit)
        A_pred_winchance = 1.0/(1.0 + 10.0**A_pred_logit)

        # update scores
        if coinflip(A_true_winchance):
            for i in range(len(teamA)):
                PUB_ELOS[teamA[i].id] += uncertainty*(1 - A_pred_winchance)
                PUB_ELOS[teamB[i].id] += uncertainty*(A_pred_winchance - 1)
        else:
            for i in range(len(teamA)):
                PUB_ELOS[teamA[i].id] += uncertainty*(-A_pred_winchance)
                PUB_ELOS[teamB[i].id] += uncertainty*(A_pred_winchance)

def update_player_change_rates(players, uncertainty):
    hist, binedges = np.histogram(PUB_ELOS, bins=32)
    players.sort()
    binedges[len(binedges) - 1] += 1000

    edge = 1
    avg_tru_elo = [0]
    avg_pub_elo = [0]
    count = 0
    for i in range(len(players)):
        if PUB_ELOS[players[i].id] < binedges[edge]:
            count += 1
            if len(avg_tru_elo) < edge:
                avg_tru_elo.append(players[i].elo_tru)
                avg_pub_elo.append(PUB_ELOS[players[i].id])
            else:
                avg_tru_elo[edge - 1] += players[i].elo_tru
                avg_pub_elo[edge - 1] += PUB_ELOS[players[i].id]
        else:
            if count != 0:
                avg_tru_elo[edge - 1] /= count
                avg_pub_elo[edge - 1] /= count
            edge += 1
            count = 0
            i -= 1
            if PUB_ELOS[players[i].id] < binedges[edge]:
                count = 1
                avg_tru_elo.append(players[i].elo_tru)
                avg_pub_elo.append(PUB_ELOS[players[i].id])
            else:
                avg_tru_elo.append(0)
                avg_pub_elo.append(0)
        if i + 1 == len(players):
            avg_tru_elo[edge - 1] /= count
            avg_pub_elo[edge - 1] /= count

    edge = 1
    for i in range(len(players)):
        if PUB_ELOS[players[i].id] > binedges[edge]:
            edge += 1
        players[i].add_change(avg_pub_elo[edge - 1], avg_tru_elo[edge - 1], players, uncertainty)
    return binedges

def coinflip(win_chance=0.5):
    if np.random.uniform(0, 1) > win_chance:
        return False
    return True

# split the playerlist into two approximately equal sub-lists (uses greedy approach)
def maketeams(playerlist):
    teamA = [playerlist[len(playerlist) - 1]]
    teamB = []
    eloA = PUB_ELOS[teamA[0].id]
    eloB = 0

    for i in range(1, len(playerlist)):
        player = playerlist[len(playerlist) - i - 1]
        if eloA > eloB:
            teamB.append(player)
            eloB += PUB_ELOS[player.id]
        else:
            teamA.append(player)
            eloA += PUB_ELOS[player.id]
    return teamA, teamB

def init_players(playercount=10000):
    f = open('data/chess_ratings.pkl','rb')
    ratings = pickle.load(f)

    # generate normal spread of scores
    gen_ratings = np.random.normal(loc=np.mean(ratings), scale=np.std(ratings), size=playercount) #np.array(ratings)
    gen_ratings.sort()

    # initialize the player base
    players = []
    for rating in gen_ratings:
        players.append(Player(rating))
    return players

if __name__=="__main__":
    main()
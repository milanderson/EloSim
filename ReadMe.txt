Usage:

------------init_players()------------
    Returns a list of players randomly initialized based on a Gaussian approximation of FIDE chess rankings
	@playercount	    Opt, default 10000


------------play_tournament(players, ...)------------
    Matches all provided players, and adjusts their estimated scores based on match results
        @players            the list of players
        @team_size          Opt, default 1. The number of players on a team
        @uncertainty        Opt, default 32. The adjustment rate. equivalent to the amount of points won/lost per match
        @learning_rate      Opt, default 1.0. The rate of decay on the uncertainty
        @rounds             Opt, default 20. The number of matches to play
        @match_rnd          Opt, default 1.0. The randomness of the matchmaking, must be a value from 1 to 0
        @usebrackets        Opt, default False. Whether to use brackets (players match players loosely within their bracket)
        @show_elo_every     Opt, default False. How often to draw the estimated Elo distribution. If None, nothing is shown
        @show_dist_every    Opt, default False. How often to draw the estimated versus real Elo. If None, nothing is shown
        @show_loss_every    Opt, default False. How often to draw the average turns to correct rank. If None, nothing is shown
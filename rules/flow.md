```mermaid
graph TD
    subgraph Legend
        direction LR
        L1(Start / End)
        L2[Process / Action]
        L3{Decision}
    end

    Start(Start Player's Turn) --> A{Has cards to play?};
    A -- No --> GameOver("GAME OVER<br/>(Cannot play or yield)");

    A -- Yes --> B{Choose Action:<br>Play Card or Yield?};

    B -- Play Card --> C[STEP 1: Play Card(s)<br>Single, Combo, or with Animal Companion];
    C --> D{Was a Jester played?};
    D -- Yes --> Jester[Jester negates enemy immunity.<br>Player who played Jester chooses next player.];
    Jester --> TurnEnd_NextPlayerChosen(End Turn);
    TurnEnd_NextPlayerChosen --> Start;

    D -- No --> E[STEP 2: Activate Suit Power(s)<br>(Immunity may apply)];
    E --> F[STEP 3: Deal Damage to Enemy<br>(Clubs count double)];
    F --> G{Is Enemy Defeated?<br>(Damage >= Health)};

    G -- No --> H[STEP 4: Suffer Damage from Enemy];

    G -- Yes --> I[Enemy Defeated!<br>Place played cards in discard.<br>Place enemy card in discard<br>or on top of Tavern deck if damage was exact.];
    I --> J{Is this the last King?};
    J -- Yes --> Win(PLAYERS WIN!);
    J -- No --> K[Reveal next Enemy from Castle Deck];
    K --> StartSamePlayer(Same Player starts a new turn);
    StartSamePlayer --> A;

    B -- Yield --> L{Can player legally yield?<br>(Not all other players yielded since your last turn)};
    L -- No --> C;
    L -- Yes --> H;

    H --> M{Can discard cards to match<br>or exceed enemy's attack value?};
    M -- Yes --> TurnEnd_Clockwise(End Turn);
    TurnEnd_Clockwise --> NextPlayer(Next Player in clockwise order<br>begins their turn);
    NextPlayer --> Start;
    M -- No --> GameOver;

    classDef default fill:#fff,stroke:#333,stroke-width:2px;
    classDef startend fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#f2d0a9,stroke:#333,stroke-width:2px;
    classDef decision fill:#bde0fe,stroke:#333,stroke-width:2px;

    class Start,StartSamePlayer,NextPlayer,TurnEnd_Clockwise,TurnEnd_NextPlayerChosen,Win,GameOver startend;
    class C,E,F,H,I,Jester,K process;
    class A,B,D,G,L,M,J decision;
    class L1,L2,L3 default;
```
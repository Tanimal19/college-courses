
b11902038 資工三 鄭博允

# How to use
## Environment
Python >= 3.13
## Run
Run the following command in the project's root directory:  
(windows) `python src/main.py`  
(linux) `./run.sh`  
## Testing
You need to install `pytest` package to perform testing!  
Run `pytest` in the project's root directory.

# Implementation Details
## Project Structure
```
b11902039_assignment01/
├── src/
│   ├── cli.py
│   ├── database.py
│   └── shop_service.py
├── test/
│   └── test_shop_service.py
├── cloudshop.db
├── pyproject.toml
├── README.md
└── run.sh
```

`database.py`: provides database to the API service, cache for top categories is also implemented here.  
`shop_service.py`: API service.  
`cli.py`: CLI interface that utilizes the API service.  
`cloudshop.db`: generated after running the app for the first time, the app will keep using the same database file unless you delete it manually.

## Database Schema
using Python's `sqlite3` module.
``` sql
TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL
);
TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    listing_count INTEGER DEFAULT 0
);
TABLE listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    price REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    username TEXT NOT NULL,
    category TEXT NOT NULL,
    FOREIGN KEY (username) REFERENCES users (name) ON DELETE CASCADE,
    FOREIGN KEY (category) REFERENCES categories (name) ON DELETE CASCADE
);
```



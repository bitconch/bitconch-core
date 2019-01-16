rustelo-easy

Rust codes written in C style, which can be easily called by go

Several things needed to be considered when write the rust codes:


- 1  For each rust file, there should be an corresponding c header file.


- 2  File structure
```
      |-----rustelo_easy     <-- The root
         |-----include       <-- The corresponding header files which will be copied to rustelo folder
         |-----src           <-- Source files in rust    
         |-----target        <-- Build output.

```

- 3 To remove a submodule you need to:

    - Delete the relevant section from the .gitmodules file.
    - Stage the .gitmodules changes git add .gitmodules
    - Delete the relevant section from .git/config.
    - Run git rm --cached path_to_submodule (no trailing slash).
    - Run rm -rf .git/modules/path_to_submodule (no trailing slash).
    - Commit git commit -m "Removed submodule <name>"
    - Delete the now untracked submodule files rm -rf path_to_submodule

- 4 SSH key issue/Access denied problems:

    - Some times, you may face the SSH issue or access denied problem when using github to add submodules.

The following tricks can be useful, from https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/#adding-your-ssh-key-to-the-ssh-agent

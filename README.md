# Thesis Template

## Mirroring this repository
Open Terminal.

Create a bare clone of the repository.

  >> $ git clone --bare https://github.com/kdd-lab/thesis_template.git

Mirror-push to the new repository.

  >> $ cd thesis_template.git
  >> $ git push --mirror https://github.com/kdd-lab/new_repository_name.git

Remove the temporary local repository you created in step 1.

  >> $ cd ..
  >> $ rm -rf thesis_template.git
  
Where ``new_repository_name`` has the following format: year_surname

- **year**: year of creation
- **sourname**: student surname

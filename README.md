# Baulab Bounties

Do you want to get involved in the exciting ongoing work at [Baulab](https://baulab.info)? Here's how.

We post open research and engineering tasks as **bounties** — bite-sized, well-defined problems drawn from our active projects. Anyone can pick one up, work on it, and submit a solution. It's the best way to demonstrate your skills and start collaborating with us.

## How It Works

### 1. Browse Open Bounties

Head to the [Issues](https://github.com/thebaulab/bounties/issues) tab. Each issue is a bounty describing a specific task — it will include context, requirements, and any relevant resources. Bounties are labeled by type (research, engineering, etc.) and difficulty.

### 2. Claim a Bounty

Found one that interests you? Leave a comment on the issue to let others know you're working on it, then fork this repo and get started.

### 3. Submit Your Work

When you're done, open a **Pull Request** that adds your submission to the `submissions/` folder. Structure it like this:

```
submissions/
  <issue-number>-<your-github-username>/
    README.md        # Summary of your approach and findings
    ...              # Code, notebooks, results, etc.
```

For example, if you're solving issue #5 and your GitHub username is `alice`:

```
submissions/
  5-alice/
    README.md
    solution.py
```

Your submission `README.md` should include:
- A brief description of your approach
- Instructions to reproduce your results
- Any key findings or takeaways

### 4. Review

A lab member will review your PR, leave feedback, and either request changes or merge it. Strong submissions are a direct path to further collaboration with the lab.

## Guidelines

- **One submission per person per bounty.** You can update your PR as many times as you like before review.
- **Work independently.** Submissions should be your own work. Using tools and libraries is fine — just document what you used.
- **Keep it clean.** Don't commit large data files, model weights, or credentials. Use `.gitignore` or link to external storage if needed.
- **Ask questions.** If a bounty's description is unclear, ask in the issue thread.

## For Lab Members

To create a new bounty, [open an issue](https://github.com/thebaulab/bounties/issues/new) with:
- A clear title and description of the task
- Relevant background, links, and references
- Acceptance criteria — what does a good submission look like?
- Appropriate labels (e.g., `research`, `engineering`)

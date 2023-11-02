import csv
import itertools
import sys

PROBS = {
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99},
    },
    "mutation": 0.01,
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }
    names = set(people)

    for have_trait in powerset(names):
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    probability = 1.0
    for person in people:
        if person in two_genes:
            num_genes = 2
        elif person in one_gene:
            num_genes = 1
        else:
            num_genes = 0
        prob_gene = PROBS["gene"][num_genes]
        prob_trait = PROBS["trait"][num_genes][person in have_trait]
        probability *= prob_gene * prob_trait
    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        num_genes = 0
        if person in two_genes:
            num_genes = 2
        elif person in one_gene:
            num_genes = 1
        probabilities[person]["gene"][num_genes] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    for person in probabilities:
        gene_dist = probabilities[person]["gene"]
        total_gene_prob = sum(gene_dist.values())
        probabilities[person]["gene"] = {
            num_genes: prob / total_gene_prob for num_genes, prob in gene_dist.items()
        }
        trait_dist = probabilities[person]["trait"]
        total_trait_prob = sum(trait_dist.values())
        probabilities[person]["trait"] = {
            has_trait: prob / total_trait_prob for has_trait, prob in trait_dist.items()
        }


if __name__ == "__main__":
    main()

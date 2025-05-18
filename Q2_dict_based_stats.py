def user_stats(data):
    stats = {}
    for user, scores in data.items():
        stats[user] = {
            "Average": sum(scores) / len(scores),
            "Min": min(scores),
            "Max": max(scores),
        }
    return stats

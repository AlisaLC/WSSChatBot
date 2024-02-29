from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, count=3, period=60):
        self.users = set()
        self.counts = {}
        self.last_access = {}
        self.period = period
        self.limit = count

    def access(self, user):
        if user not in self.users:
            self.users.add(user)
            self.counts[user] = 1
            self.last_access[user] = datetime.now()
            return True
        now = datetime.now()
        delta = now - self.last_access[user]
        count = max(self.counts[user] - delta.seconds * self.limit / self.period, 0)
        if count < self.limit:
            return True
        return False
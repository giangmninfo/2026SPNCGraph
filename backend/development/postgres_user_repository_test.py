from backend.domain.entities.user import User
from backend.infrastructure.repositories.postgres_user_repository import PostgresUserRepository


def main():
    repo = PostgresUserRepository()

    user = User(
        id=None,
        full_name="Test User",
        username="testuser_2",
        email="testuser2@example.com",
        password=b"plaintext-for-now",
        avatar_color="#ffcc00",
        created_at=None,
    )

    created = repo.create(user)

    print("CREATED:")
    print(created.__dict__)

    fetched = repo.get_by_id(created.id)
    print("\nFETCHED BY ID:")
    print(fetched.__dict__)

    fetched_by_username = repo.get_by_username("testuser_1")
    print("\nFETCHED BY USERNAME:")
    print(fetched_by_username.__dict__)

    all_users = repo.list_all()
    print(f"\nTOTAL USERS: {len(all_users)}")

    repo.close()


if __name__ == "__main__":
    main()

from sqlalchemy import text
from db.database import engine
import pandas as pd
from typing import List


def get_table_names_by_user_id(user_id: str) -> List[str]:
    query = text("SELECT DISTINCT table_name FROM experiments WHERE user_id = :user_id")
    with engine.connect() as conn:
        result = conn.execute(query, {"user_id": user_id})
        conn.commit()
        conn.close()
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df["table_name"].tolist()


def create_experiments_table() -> None:
    # Create the profiles table if it doesn't exist
    create_table_stmt = """
    CREATE TABLE IF NOT EXISTS experiments (
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES auth.users,
        table_name TEXT,
        csv_dict JSONB,
        columns_order JSON
    );
    """
    with engine.connect() as conn:
        query = text(create_table_stmt)
        conn.execute(query)
        conn.commit()
        conn.close()


def enable_rls_for_table(table_name: str):
    table_name = table_name.lower()
    # Enable RLS
    with engine.connect() as conn:
        query = text(
            f"""
            ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;
            """
        )
        conn.execute(query)
        conn.commit()
        print(f"RLS for table {table_name} created in database")
        conn.close()


def create_policy_for_table(table_name: str):
    table_name = table_name.lower()
    # Enable RLS
    with engine.connect() as conn:
        query = text(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_policies
                    WHERE schemaname = 'public'
                    AND tablename = '{table_name}'
                    AND policyname = 'User CRUD own tables only'
                ) THEN
                    CREATE POLICY "User CRUD own tables only"
                    ON public.{table_name}
                    FOR ALL
                    TO authenticated
                    USING (
                        auth.uid() = user_id
                    )
                    WITH CHECK (
                        auth.uid() = user_id
                    );
                END IF;
            END
            $$;
            """
        )
        conn.execute(query)
        conn.commit()
        print(f"Created policy for table {table_name}.")
        conn.close()

export async function createUser(payload: {
    full_name: string
    username: string
    email: string
    password: string
    avatar_color: string
  }) {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_API_URL}/users`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      }
    )
  
    const data = await res.json()
  
    if (!res.ok) {
      throw new Error(data.error || "Signup failed")
    }
  
    return data
  }
  
export async function checkUsername(username: string) {
  const res = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/users/availability?username=${username}`
  )
  if (!res.ok) throw new Error("Failed")
  return res.json()
}
'use client'

// import { useState, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { X, Search, ChevronLeft, ChevronRight } from 'lucide-react'
import { GitHubAvatar, getAvatarColorForUser } from "@/components/github-avatar"
import type { UIClassificationHistory } from "@/lib/api/history.mapper"
import { useEffect, useState, useMemo } from "react"
import { fetchClassificationHistory } from "@/lib/api/model"
import { mapHistoryItemToUI } from "@/lib/api/history.mapper"

interface ProfileModalProps {
  isOpen: boolean
  onClose: () => void
  onLogout: () => void
  user?: {
    fullName: string
    username: string
    email: string
  }
}

export function ProfileModal({ isOpen, onClose, onLogout, user }: ProfileModalProps) {
  const [history, setHistory] = useState<UIClassificationHistory[]>([])
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [searchTerm, setSearchTerm] = useState("")
  const [loading, setLoading] = useState(false)

  const avatarColor = user
    ? getAvatarColorForUser(user.username)
    : "#4ECDC4"

  useEffect(() => {
    if (!isOpen || !user) return

    setLoading(true)

    fetchClassificationHistory({
      page: currentPage,
      q: searchTerm || undefined,
    })
      .then((res) => {
        setHistory(res.items.map(mapHistoryItemToUI))
        setTotalPages(res.total_pages)
      })
      .finally(() => setLoading(false))
  }, [isOpen, user, currentPage, searchTerm])

  if (!isOpen || !user) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="modal-bg border border-border bg-background rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border flex-shrink-0">
          <h2 className="text-2xl font-semibold text-foreground">User Profile</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto flex-1 [&::-webkit-scrollbar]:hidden">
          {/* User Info Section */}
          <div className="flex items-center gap-6">
            <GitHubAvatar username={user.username} size={120} />
            <div className="space-y-4">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Full Name</p>
                <p className="font-semibold text-lg" style={{ color: avatarColor }}>{user.fullName}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Username</p>
                <p className="font-medium text-lg" style={{ color: avatarColor }}>@{user.username}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Email</p>
                <p className="font-medium text-lg" style={{ color: avatarColor }}>{user.email}</p>
              </div>
            </div>
          </div>

          {/* History Section */}
          <div className="border-t border-border pt-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
              <h3 className="font-semibold text-lg text-foreground">CLASSIFICATION HISTORY</h3>
              
              {/* Search Bar */}
              <div className="relative w-full md:w-72">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search by ID or Subject..."
                  className="w-full pl-9 pr-4 py-2 bg-muted/50 border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/20"
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value)
                    setCurrentPage(1)
                  }}
                />
              </div>
            </div>

            <div className="space-y-3 min-h-[200px]">
              {history.length > 0 ? (
                history.map((item) => (
                  <Card key={item.id} className="p-3 hover:border-primary/50 transition-colors bg-card">
                    <div className="flex gap-4 items-start">
                      <img
                        src={item.image || "/placeholder.svg"}
                        alt={item.subject}
                        className="w-20 h-20 rounded object-cover flex-shrink-0"
                      />
              <div className="flex-1 grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                {/* ID */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    ID
                  </p>
                  <p className="font-medium text-foreground">
                    {item.id}
                  </p>
                </div>

                {/* Subject */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    Subject
                  </p>
                  <p className="font-medium text-foreground">
                    {item.subject}
                  </p>
                </div>

                {/* Grade — NEW */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    Grade
                  </p>
                  <p className="font-medium text-foreground">
                    {item.gradeLabel}
                  </p>
                </div>

                {/* Confidence */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    Confidence
                  </p>
                  <p className="font-medium text-green-600">
                    {item.confidenceLabel}
                  </p>
                </div>

                {/* Model — MOVED DOWN */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    Model
                  </p>
                  <p className="font-medium text-foreground">
                    {item.modelVariant}
                  </p>
                </div>

                {/* Created Date */}
                <div>
                  <p className="text-xs font-semibold text-muted-foreground uppercase">
                    Created Date
                  </p>
                  <p className="font-medium text-foreground">
                    {item.createdDate}
                  </p>
                </div>
              </div>
              </div>
                  </Card>
                ))
              ) : (
                <div className="text-center py-10 text-muted-foreground">No history found.</div>
              )}
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-4 mt-6">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="w-4 h-4" /> Previous
                </Button>

                <span className="text-sm font-medium">
                  Page {currentPage} of {totalPages}
                </span>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(p) => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages}
                  className="flex items-center gap-1"
                >
                  Next <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="border-t border-border p-6 flex gap-3 flex-shrink-0">
          <Button onClick={onClose} variant="outline" className="flex-1">
            Close
          </Button>
          <Button 
            onClick={onLogout} 
            variant="outline"
            className="flex-1"
            style={{ backgroundColor: avatarColor }}
          >
            Logout
          </Button>
        </div>
      </div>
    </div>
  )
}
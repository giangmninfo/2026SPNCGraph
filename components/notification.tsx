'use client'

import { useState, useRef, useEffect } from 'react'
import { X, AlertCircle, CheckCircle, Info } from 'lucide-react'

export type NotificationType = 'success' | 'error' | 'info' | 'warning'

interface Notification {
  id: string
  message: string
  type: NotificationType
  duration?: number
}

export function useNotification() {
  const [notifications, setNotifications] = useState<Notification[]>([])

  const addNotification = (
    message: string,
    type: NotificationType = 'info',
    duration = 2000
  ) => {
    const id = Math.random().toString(36).substr(2, 9)
    setNotifications((prev) => [...prev, { id, message, type, duration }])
  }

  const removeNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id))
  }

  return { notifications, addNotification, removeNotification }
}

export function NotificationContainer({
  notifications,
  onRemove,
}: {
  notifications: Notification[]
  onRemove: (id: string) => void
}) {
  const [animatingIds, setAnimatingIds] = useState<Set<string>>(new Set())

  const autoTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const hoverTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

  useEffect(() => {
    // start auto-dismiss timers for notifications that don't already have timers
    notifications.forEach((n) => {
      if (
        n.duration &&
        n.duration > 0 &&
        !autoTimersRef.current.has(n.id) &&
        !hoverTimersRef.current.has(n.id)
      ) {
        const t = setTimeout(() => {
          setAnimatingIds((prev) => new Set(prev).add(n.id))
          // allow CSS animation to run before removing
          setTimeout(() => {
            onRemove(n.id)
            setAnimatingIds((prev) => {
              const s = new Set(prev)
              s.delete(n.id)
              return s
            })
            autoTimersRef.current.delete(n.id)
          }, 300)
        }, n.duration)
        autoTimersRef.current.set(n.id, t)
      }
    })

    // cleanup timers for notifications that no longer exist
    const ids = new Set(notifications.map((n) => n.id))
    Array.from(autoTimersRef.current.keys()).forEach((id) => {
      if (!ids.has(id)) {
        const t = autoTimersRef.current.get(id)
        if (t) clearTimeout(t)
        autoTimersRef.current.delete(id)
      }
    })
    Array.from(hoverTimersRef.current.keys()).forEach((id) => {
      if (!ids.has(id)) {
        const t = hoverTimersRef.current.get(id)
        if (t) clearTimeout(t)
        hoverTimersRef.current.delete(id)
      }
    })

    return () => {
      // on unmount clear timers
      Array.from(autoTimersRef.current.values()).forEach((t) => clearTimeout(t))
      Array.from(hoverTimersRef.current.values()).forEach((t) => clearTimeout(t))
      autoTimersRef.current.clear()
      hoverTimersRef.current.clear()
    }
  }, [notifications, onRemove])

  const handleMouseEnter = (id: string) => {
    // cancel any auto-dismiss timer
    const auto = autoTimersRef.current.get(id)
    if (auto) {
      clearTimeout(auto)
      autoTimersRef.current.delete(id)
    }

    // clear existing hover timer then start a 5s hover timer that will animate out
    const existing = hoverTimersRef.current.get(id)
    if (existing) clearTimeout(existing)

    const ht = setTimeout(() => {
      setAnimatingIds((prev) => new Set(prev).add(id))
      setTimeout(() => {
        onRemove(id)
        setAnimatingIds((prev) => {
          const s = new Set(prev)
          s.delete(id)
          return s
        })
        hoverTimersRef.current.delete(id)
      }, 300)
    }, 3000)

    hoverTimersRef.current.set(id, ht)
  }

  const handleMouseLeave = (id: string) => {
    // cancel hover timer
    const ht = hoverTimersRef.current.get(id)
    if (ht) {
      clearTimeout(ht)
      hoverTimersRef.current.delete(id)
    }

    // restart auto-dismiss if notification has a duration
    const n = notifications.find((x) => x.id === id)
    if (n && n.duration && n.duration > 0 && !autoTimersRef.current.has(id)) {
      const t = setTimeout(() => {
        setAnimatingIds((prev) => new Set(prev).add(id))
        setTimeout(() => {
          onRemove(id)
          setAnimatingIds((prev) => {
            const s = new Set(prev)
            s.delete(id)
            return s
          })
          autoTimersRef.current.delete(id)
        }, 300)
      }, n.duration)
      autoTimersRef.current.set(id, t)
    }
  }

  return (
    <div className="fixed top-4 right-4 z-[999] space-y-2 max-w-sm">
      {notifications.map((notification) => {
        const isAnimating = animatingIds.has(notification.id)

        const accentClass =
          notification.type === 'success'
            ? 'w-1.5 bg-green-700 dark:bg-green-300 rounded-l-md'
            : notification.type === 'error'
            ? 'w-1.5 bg-red-700 dark:bg-red-300 rounded-l-md'
            : notification.type === 'warning'
            ? 'w-1.5 bg-yellow-700 dark:bg-yellow-300 rounded-l-md'
            : 'w-1.5 bg-blue-700 dark:bg-blue-300 rounded-l-md'

        const contentBase = 'flex items-center gap-3 px-4 py-3 flex-1 w-full border rounded-r-lg'

        const contentVariant =
          notification.type === 'success'
            ? 'bg-white border-green-100 text-green-800 dark:bg-green-950 dark:text-green-200 dark:border-green-800'
            : notification.type === 'error'
            ? 'bg-white border-red-100 text-red-800 dark:bg-red-950 dark:text-red-200 dark:border-red-800'
            : notification.type === 'warning'
            ? 'bg-white border-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-200 dark:border-yellow-800'
            : 'bg-white border-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-200 dark:border-blue-800'

        const iconClass =
          notification.type === 'success'
            ? 'w-5 h-5 flex-shrink-0 text-green-700'
            : notification.type === 'error'
            ? 'w-5 h-5 flex-shrink-0 text-red-700 dark:text-red-200'
            : notification.type === 'warning'
            ? 'w-5 h-5 flex-shrink-0 text-yellow-700 dark:text-yellow-200'
            : 'w-5 h-5 flex-shrink-0 text-blue-700 dark:text-blue-200'

        return (
          <div
            key={notification.id}
            className={`flex overflow-hidden rounded-lg shadow-lg transition-all duration-300 ${
              isAnimating ? 'translate-x-96 opacity-0 pointer-events-none' : 'translate-x-0 opacity-100'
            }`}
            onMouseEnter={() => handleMouseEnter(notification.id)}
            onMouseLeave={() => handleMouseLeave(notification.id)}
          >
            <div className={accentClass} />

            <div className={`${contentBase} ${contentVariant}`}>
              {notification.type === 'success' && <CheckCircle className={iconClass} />}
              {notification.type === 'error' && <AlertCircle className={iconClass} />}
              {notification.type === 'warning' && <AlertCircle className={iconClass} />}
              {notification.type === 'info' && <Info className={iconClass} />}

              <span className="text-sm font-medium flex-1">{notification.message}</span>

              <button
                onClick={() => onRemove(notification.id)}
                className="text-current hover:opacity-70 transition-opacity flex-shrink-0"
                aria-label="Dismiss notification"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
